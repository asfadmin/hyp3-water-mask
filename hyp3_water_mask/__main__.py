"""
water_mask processing for HyP3
"""

import os
import re
import traceback
from datetime import datetime

import boto3
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
from keras.models import load_model as kload_model
from osgeo import gdal

import hyp3_water_mask


class NoVVVHError(Exception):
    log.info("Error: RTCs must have 'VV' or 'VH' in their filenames to run a water mask.")
    pass

def download_product(cfg, url):
    cmd = ('wget -nc %s -P %s' % (url, cfg['workdir']))

    o, ok = get_asf.execute(cmd, quiet=True)
    if not ok:
        log.info('Failed to download HyP3 product: {0}'.format(url))
        return False

    file_name = os.path.basename(url)
    zip_file = os.path.join(cfg['workdir'], file_name)

    if not os.path.isfile(zip_file):
        msg = "An error occurred while preparing the products for processing."
        failure(cfg, msg)
        msg = "Could not find expected download file: {0}"
        raise Exception(msg.format(zip_file))
    else:
        log.info('Download complete')
        log.info('Unzipping {0}'.format(zip_file))

        unzip(zip_file, cfg['workdir'])

        log.info('Unzip completed.')
        return True


# FIXME: copied in from proc_lib on the afl_dev branch... possibly bring into
#        hyp3proclib
def download_from_s3(src, dst, cfg, bucket):

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
        region_name=cfg["aws_region"],
    )
    s3_client.download_file(bucket, src, dst)


def get_tile_row_col_count(height, width, tile_size):
    return int(np.ceil(height / tile_size)), int(np.ceil(width / tile_size))


def pad_image(image, to):
    height, width = image.shape

    n_rows, n_cols = get_tile_row_col_count(height, width, to)
    new_height = n_rows * to
    new_width = n_cols * to

    padded = np.zeros((new_height, new_width))
    padded[:image.shape[0], :image.shape[1]] = image
    return padded


def tile_image(image, width=512, height=512):
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


def write_mask_to_file(mask, file_name, projection, geo_transform):
    (width, height) = mask.shape
    out_image = gdal.GetDriverByName('GTiff').Create(
        file_name, height, width, bands=1
    )
    out_image.SetProjection(projection)
    out_image.SetGeoTransform(geo_transform)
    out_image.GetRasterBand(1).WriteArray(mask)
    out_image.GetRasterBand(1).SetNoDataValue(0)
    out_image.FlushCache()


def mask_img(product, model, dst):
    extract_name = re.compile(r"(.*).zip")        # Make sure this works
    sar_regex = re.compile(r"(.*)_(VH|VV).tif")   # Make sure this works

    mask = None
    f = None
    invalid_pixels = None

    # Needed information for making a water mask
    # Subscription ID
    # Path . . . probably going to be a path to a S3 bucket

    m = re.match(extract_name, product)
    folder = m.groups()
    log.info("DEBUGGING: folder: {}".format(folder))
    for gran in os.listdir(folder[0]):
        granule = os.path.join(folder[0], gran)
        m = re.match(sar_regex, gran)
        if not m or gran.endswith('xml'):
            continue

        _, band = m.groups()

        f = gdal.Open(granule)
        img_array = f.ReadAsArray()
        original_shape = img_array.shape
        n_rows, n_cols = get_tile_row_col_count(*original_shape, tile_size=512)

        vh_tiles = None
        vv_tiles = None

        if band == "VH":
            vh_array = pad_image(f.ReadAsArray(), 512)
            invalid_pixels = np.nonzero(vh_array == 0.0)
            vh_tiles = tile_image(vh_array)
        elif band == "VV":
            vv_array = pad_image(f.ReadAsArray(), 512)
            invalid_pixels = np.nonzero(vv_array == 0.0)
            vv_tiles = tile_image(vv_array)
        else:
            log.info("Cannot run a water mask on {}".format(granule))
            raise NoVVVHError
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
    outfile = "{}/{}_WM.tif".format(dst, folder)
    write_mask_to_file(mask, outfile, f.GetProjection(), f.GetGeoTransform())


def load_model(model_path):
    """ Loads and returns a model. Attaches the model name and that model's
    history. """
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    log.info("model_dir: {}".format(model_dir))
    model = kload_model(model_path)

    # Attach our extra data to the model
    model.__asf_model_name = model_name

    return model


def process_water_mask(cfg, n):
    bucket = "s3://asf-ai-water-training-data/Networks/"
    model_src = "new64.zip"
    products = get_extra_arg(cfg, 'hyp3Products', '')
    input_type = cfg['input_type']

    if input_type.lower() == 'rtc':
        input_type = 'RTC'
    else:
        failure(cfg, "Something went wrong, input type should be RTC.")
        raise Exception("Something went wrong, input type should be RTC.")

    message = "Processing water mask(s) from subscription {0} for {1}"
    log.info(message.format(cfg['sub_name'], cfg['username']))

    # Start downloading and processing
    date_time = str(datetime.now())
    cfg['log'] = "Processing started at {0} \n\n".format(date_time)
    download_count = 0
    # load model
    try:
        download_from_s3(model_src, model_src, cfg, bucket)
    except Exception:
        log.info("Unexpected error: {}".format(traceback.format_exc()))
        raise

    model_dir = 'model'
    unzip(model_src, model_dir)
    model = load_model("{}/latest.h5".format(model_dir))

    output_path = "{}_water_masks".format(cfg['sub_id'])
    os.mkdir(output_path)
    for product_url in products:
        if download_product(cfg, product_url):
            download_count += 1
            # Mask the product
            product_name, _ = os.path.splitext(product_url.split('/')[-1])
            log.info("Creating a water mask for {}".format(product_name))
            mask_img(product_url, model, output_path)

    if download_count == 0:
        msg = "No {0} products for this job could be found. Are they expired?"
        failure(cfg, msg.format(input_type))
        msg = "No {0} products downloaded to process."
        raise Exception(msg.format(input_type))

    with get_db_connection('hyp3-db') as conn:
        log.debug("Adding citation and zipping folder at {0}".format(output_path))
        add_citation(cfg, output_path)
        zip_file = "{}.zip".format(output_path)
        zip_dir(output_path, zip_file)

        if 'lag' in cfg and 'email_text' in cfg:
            cfg['email_text'] += "\n" + "You are receiving this product {0} after it was acquired.".format(cfg['lag'])

        cfg['final_product_size'] = [os.stat(zip_file).st_size, ]
        cfg['original_product_size'] = 0
        record_metrics(cfg, conn)

        log.debug("Using file paths: {0}".format(zip_file))
        upload_product(zip_file, cfg, conn)
        success(conn, cfg)


def main():
    """
    Main entrypoint for hyp3_water_mask
    """
    processor = Processor(
        'water_mask', hyp3_water_mask, sci_version=hyp3_water_mask.__version__
    )
    processor.run()


if __name__ == '__main__':
    main()
