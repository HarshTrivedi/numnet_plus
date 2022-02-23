FROM python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /local

RUN pip install allennlp==0.9.0
RUN pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -U scikit-learn
RUN pip install overrides==3.1.0

# Copy remaining code.
COPY numnet_plus/drop_eval.py drop_eval.py
COPY numnet_plus/eval.sh eval.sh
COPY numnet_plus/mspan_roberta_gcn mspan_roberta_gcn
COPY numnet_plus/numnet_plus_pic.png numnet_plus_pic.png
COPY numnet_plus/options.py options.py
COPY numnet_plus/prepare_roberta_data.py prepare_roberta_data.py
COPY numnet_plus/roberta_gcn_cli.py roberta_gcn_cli.py
COPY numnet_plus/roberta_predict.py roberta_predict.py
COPY numnet_plus/tag_mspan_robert_gcn tag_mspan_robert_gcn
COPY numnet_plus/tools tools
COPY numnet_plus/train.sh train.sh
COPY numnet_plus/train_beaker.sh train_beaker.sh
COPY numnet_plus/train_on_beaker.py train_on_beaker.py

CMD ["/bin/bash"]

