"""
water_mask processing for HyP3
"""

import os
import re
from datetime import datetime
from glob import glob

import numpy as np
from hyp3lib import get_asf
from hyp3proclib import (
    failure,
    get_extra_arg,
    record_metrics,
    success,
    unzip,
    upload_product,
    zip_dir
)
from hyp3proclib.db import get_db_connection
from hyp3proclib.file_system import add_citation
from hyp3proclib.logger import log
from hyp3proclib.proc_base import Processor
from keras import Model as keras_model
from keras.models import load_model as kload_model
from osgeo import gdal

import hyp3_water_mask


class NoVVVHError(Exception):
    log.info("Error: RTCs must have 'VV' or 'VH'"
             "in their filenames to run a water mask.")
    pass


def download_product(cfg: dict, url: str, download_dir: str) -> bool:
    cmd = ('wget -nc %s -P %s' % (url, f"{download_dir}"))
    log.info(f"wget command: {cmd}")
    o, ok = get_asf.execute(cmd, quiet=True)
    if not ok:
        log.info('Failed to download HyP3 product: {0}'.format(url))
        return False

    file_name = os.path.basename(url)
    zip_file = os.path.join(download_dir, file_name)

    if not os.path.isfile(zip_file):
        msg = "An error occurred while preparing the products for processing."
        failure(cfg, msg)
        msg = "Could not find expected download file: {0}"
        raise Exception(msg.format(zip_file))
    else:
        log.info('Download complete')
        log.info(f"Unzipping {zip_file} to {download_dir}")
        unzip(zip_file, download_dir)
        log.info(f"download_dir ls: {os.listdir(download_dir)}")
        log.info('Unzip completed.')
        return True


def get_tile_row_col_count(height: int, width: int, tile_size: int) -> tuple:
    return int(np.ceil(height / tile_size)), int(np.ceil(width / tile_size))


def pad_image(image: np.ndarray, to: int) -> np.ndarray:
    height, width = image.shape

    n_rows, n_cols = get_tile_row_col_count(height, width, to)
    new_height = n_rows * to
    new_width = n_cols * to

    padded = np.zeros((new_height, new_width))
    padded[:image.shape[0], :image.shape[1]] = image
    return padded


def tile_image(image: np.ndarray, width=512, height=512) -> np.ndarray:
    _nrows, _ncols = image.shape
    _strides = image.strides

    strides = [height * _strides[0], width * _strides[1]]
    for s in _strides:
        strides.append(s)

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)

    assert _m == 0, "Image must be evenly tileable. Please pad it first"
    assert _n == 0, "Image must be evenly tileable. Please pad it first"

    return np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width),
        strides=tuple(strides),
        writeable=False
    ).reshape(nrows * ncols, height, width)


def write_mask_to_file(mask: np.ndarray, file_name: str, projection: str, geo_transform: tuple) -> None:
    (width, height) = mask.shape
    out_image = gdal.GetDriverByName('GTiff').Create(
        file_name, height, width, bands=1
    )
    out_image.SetProjection(projection)
    out_image.SetGeoTransform(geo_transform)
    out_image.GetRasterBand(1).WriteArray(mask)
    out_image.GetRasterBand(1).SetNoDataValue(0)
    out_image.FlushCache()


def group_polarizations(tif_paths: list) -> dict:
    pths = {}
    for tif in tif_paths:
        product_name = os.path.basename(tif).split('.')[0][:-3]
        if product_name in pths:
            pths[product_name].append(tif)
        else:
            pths.update({product_name: [tif]})
            pths[product_name].sort()
    return pths


def confirm_dual_polarizations(paths: dict) -> bool:
    if len(paths) > 0:
        vv_regex = "(vv|VV)"
        vh_regex = "(vh|VH)"
        for pth in paths:
            vv = False
            vh = False
            if len(paths[pth]) == 2:
                for p in paths[pth]:
                    if re.search(vv_regex, p):
                        vv = True
                    elif re.search(vh_regex, p):
                        vh = True
                if not vv or not vh:
                    return False
            else:
                return False
        return True
    else:
        return False


def get_tif_paths(regex: str, pths: str) -> list:
    tif_paths = []
    for pth in glob(pths):
        tif_path = re.search(regex, pth)
        if tif_path:
            tif_paths.append(pth)
    return tif_paths


def load_model(model_path: str) -> keras_model:
    """ Loads and returns a model. Attaches the model name and that model's
    history. """
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    log.info("model_dir: {}".format(model_dir))
    model = kload_model(model_path)

    # Attach our extra data to the model
    model.__asf_model_name = model_name

    return model


def make_masks(grouped_paths: dict, model: keras_model, output_dir: str) -> bool:
    for pair in grouped_paths:
        for tif in grouped_paths[pair]:
            f = gdal.Open(tif)
            img_array = f.ReadAsArray()
            original_shape = img_array.shape
            n_rows, n_cols = get_tile_row_col_count(*original_shape, tile_size=512)
            log.info(f'tif: {tif}')
            if 'vv' in tif or 'VV' in tif:
                vv_array = pad_image(f.ReadAsArray(), 512)
                invalid_pixels = np.nonzero(vv_array == 0.0)
                vv_tiles = tile_image(vv_array)
            else:
                vh_array = pad_image(f.ReadAsArray(), 512)
                invalid_pixels = np.nonzero(vh_array == 0.0)
                vh_tiles = tile_image(vh_array)

        # Predict masks
        masks = model.predict(
            np.stack((vh_tiles, vv_tiles), axis=3), batch_size=1, verbose=1
        )
        masks.round(decimals=0, out=masks)
        # Stitch masks together
        mask = masks.reshape((n_rows, n_cols, 512, 512)) \
            .swapaxes(1, 2) \
            .reshape(n_rows * 512, n_cols * 512)  # yapf: disable

        mask[invalid_pixels] = 0
        filename, ext = os.path.basename(tif).split('.')
        outfile = f"{output_dir}/{filename[:-3]}_water_mask.{ext}"
        write_mask_to_file(mask, outfile, f.GetProjection(), f.GetGeoTransform())
        if not os.path.isfile(outfile):
            log.info(f"failed to write mask to {outfile}")
            return False
    return True


def process_water_mask(cfg: dict, n: int) -> None:
    log.info("process_water_mask")
    product_urls = get_extra_arg(cfg, 'hyp3Products', '')
    log.info(f"products for masking: {product_urls}")
    log.info(f"cfg: {cfg}")
    message = "Processing water mask(s) from subscription {0} for {1}"
    log.info(message.format(cfg['sub_name'], cfg['username']))

    # Start downloading and processing
    date_time = str(datetime.now())
    cfg['log'] = f"Processing started at {date_time}"
    download_count = 0

    # load model
    log.info(os.listdir())
    model = load_model("/home/conda/network.h5")
    log.info(f"model: {model}")

    workdir = os.getcwd()
    log.info(f"workdir: {workdir}")
    products_path = f"{workdir}/products"
    output_path = f"{workdir}/output"
    os.mkdir(output_path)
    os.mkdir("products")
    for product_url in product_urls:
        log.info(f"product_url: {product_url}")
        if download_product(cfg, product_url, f"{products_path}/"):
            download_count += 1
        if download_count == 0:
            msg = "No products for this job could be found. Are they expired?"
            failure(cfg, msg)
            msg = "No products downloaded to process."
            raise Exception(msg)
        else:
            log.info(f"Downloaded {download_count} product/s")

    # Identify and group VV/VH tif pairs
    product_paths = f"{products_path}/*/*"
    tif_regex = "\\w[\\--~]{5,300}(_|-)V(v|V|h|H).(tif|tiff)$"
    tif_paths = get_tif_paths(tif_regex, product_paths)
    log.info(f"tif_paths: {tif_paths}")
    grouped_paths = group_polarizations(tif_paths)
    log.info(f"grouped_paths: {grouped_paths}")
    if not confirm_dual_polarizations(grouped_paths):
        log.info("ERROR: Hyp3_water_mask requires both VV and VH polarizations.")
    else:
        log.info("Confirmed presence of VV and VH polarities for each product.")

    # Generate masks
    proc_success = make_masks(grouped_paths, model, output_path)
    log.info(f"water_masks: {os.listdir(output_path)}")

    # Upload products and update database
    with get_db_connection('hyp3-db') as conn:
        log.debug(f"Adding citation and zipping folder at {output_path}")
        add_citation(cfg, output_path)
        zip_file = f"water_mask_{cfg['sub_id']}.zip"
        log.info(f"zip_file: {zip_file}")
        zip_dir(output_path, zip_file)

        log.info(f"workdir ls: {os.listdir(workdir)}")

        if 'lag' in cfg and 'email_text' in cfg:
            cfg['email_text'] += "\nYou are receiving this product {0}" \
                                 " after it was acquired.".format(cfg['lag'])

        cfg['final_product_size'] = [os.stat(zip_file).st_size, ]
        cfg['original_product_size'] = 0
        cfg['process_time'] = str(datetime.now() - cfg['process_start_time'])
        cfg['subscriptions'] = cfg['sub_id']
        cfg['processes'] = cfg['proc_id']
        if proc_success:
            cfg['success'] = True
        else:
            cfg['success'] = False
        record_metrics(cfg, conn)

        log.debug("Using file paths: {0}".format(zip_file))
        upload_product(zip_file, cfg, conn)
        success(conn, cfg)


def main():
    """
    Main entrypoint for hyp3_water_mask
    """

    processor = Processor(
        'water_mask', process_water_mask, sci_version=hyp3_water_mask.__version__
    )
    processor.run()


if __name__ == '__main__':
    main()
