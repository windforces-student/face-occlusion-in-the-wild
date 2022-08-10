FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV prometheus_multiproc_dir="/tmp"
ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL software-properties-common && \
        add-apt-repository 'ppa:deadsnakes/ppa' && \
        apt-get update

RUN DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.7 \
        git \
        ssh \
        python3.7-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-tk \
        python3-pybind11 \
        libprotoc-dev \
        protobuf-compiler \
        curl \
        apt-transport-https \
        ca-certificates \
        gnupg \
        && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    alias pip=pip3

RUN python3 -m pip install --upgrade pip==21.3.1 setuptools==60.2.0 wheel==0.37.1

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install requirements.txt
COPY . /workspace/InsightFace_Pytorch/
RUN pip3 install --trusted-host ftp.daumkakao.com -i http://ftp.daumkakao.com/pypi/simple -r /workspace/InsightFace_Pytorch/requirements.txt

# Install ONNX-RUNTIME
WORKDIR /workspace
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update \
    && apt install cmake

RUN git clone -b v0.3.1 https://github.com/microsoft/onnxruntime.git \
    && cd onnxruntime \
    && bash build.sh --config Release --build_shared_lib --parallel --enable_pybind --build_wheel --update --build \
    && pip3 install build/Linux/Release/dist/onnxruntime-0.3.1-cp37-cp37m-linux_x86_64.whl
