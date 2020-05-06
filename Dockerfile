FROM continuumio/miniconda3:4.7.12

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="HyP3 Water Mask"
LABEL org.opencontainers.image.description="HyP3 plugin for water masking"
LABEL org.opencontainers.image.vendor="Alaska Satellite Facility"
LABEL org.opencontainers.image.authors="ASF APD/Tools Team <uaf-asf-apd@alaska.edu>"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://scm.asf.alaska.edu/hyp3/hyp3-water-mask"
LABEL org.opencontainers.image.source="https://scm.asf.alaska.edu/hyp3/hyp3-water-mask"
# LABEL org.opencontainers.image.documentation=""

# Dynamic lables to define at build time via `docker build --label`
# LABEL org.opencontainers.image.created=""
# LABEL org.opencontainers.image.version=""
# LABEL org.opencontainers.image.revision=""

RUN apt-get update && apt-get install -y unzip vim && apt-get clean

ARG CONDA_GID=1000
ARG CONDA_UID=1000

RUN groupadd -g "${CONDA_GID}" --system conda && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/conda -m  -s /bin/bash conda && \
    conda update -n base -c defaults conda && \
    chown -R conda:conda /opt && \
    conda clean -afy && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/conda/.profile && \
    echo "conda activate base" >> /home/conda/.profile

USER ${CONDA_UID}
SHELL ["/bin/bash", "-l", "-c"]
ENV PYTHONDONTWRITEBYTECODE=true
WORKDIR /home/conda/

ENV S3_PYPI_HOST="hyp3-pypi.s3-website-us-east-1.amazonaws.com"
ENV HYP3_REGISTRY="626226570674.dkr.ecr.us-east-1.amazonaws.com"
ARG SDIST_SPEC

RUN conda create -n hyp3-water-mask -c conda-forge python=3.7 boto3 gdal imageio importlib_metadata \
    keras lxml matplotlib netCDF4 numpy pillow proj \
    psycopg2 pyshp pytest pytest-console-scripts pytest-cov requests scipy \
    setuptools six statsmodels wheel && \
    conda clean -afy && \
    conda activate hyp3-water-mask && \
    sed -i 's/conda activate base/conda activate hyp3-water-mask/g' /home/conda/.profile && \
    python3 -m pip install --no-cache-dir hyp3_water_mask${SDIST_SPEC} \
    --trusted-host "${S3_PYPI_HOST}" \
    --extra-index-url "http://${S3_PYPI_HOST}"


ENTRYPOINT ["conda", "run", "-n", "hyp3-water-mask", "hyp3_water_mask"]
CMD ["-h"]

