FROM continuumio/miniconda3:4.7.12

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="HyP3 Water Mask"
LABEL org.opencontainers.image.description="HyP3 plugin for water masking"
LABEL org.opencontainers.image.vendor="Alaska Satellite Facility"
LABEL org.opencontainers.image.authors="ASF APD/Tools Team <uaf-asf-apd@alaska.edu>"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://github.com/asfadmin/hyp3-water-mask"
LABEL org.opencontainers.image.source="https://github.com/asfadmin/hyp3-water-mask"
# LABEL org.opencontainers.image.documentation=""

# Dynamic lables to define at build time via `docker build --label`
# LABEL org.opencontainers.image.created=""
# LABEL org.opencontainers.image.version=""
# LABEL org.opencontainers.image.revision=""

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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

COPY network.h5 ${WORKDIR}network.h5
COPY conda-env.yml ${WORKDIR}conda-env.yml

ARG S3_PYPI_HOST
ARG SDIST_SPEC

RUN conda env create -f ${WORKDIR}conda-env.yml && \
    conda clean -afy && \
    conda activate hyp3-water-mask && \
    sed -i 's/conda activate base/conda activate hyp3-water-mask/g' /home/conda/.profile && \
    python3 -m pip install --no-cache-dir hyp3_water_mask${SDIST_SPEC} \
    --trusted-host "${S3_PYPI_HOST}" \
    --extra-index-url "http://${S3_PYPI_HOST}"


ENTRYPOINT ["conda", "run", "-n", "hyp3-water-mask", "hyp3_water_mask"]
CMD ["-h"]

