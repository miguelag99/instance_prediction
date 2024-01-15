FROM nvidia/cuda:11.2.2-base-ubuntu20.04

# Obtain the UID and GID of the current user to create a user with the same ID, this is to avoid permission issues when mounting local volumes.
ARG USER=perception
ARG USER_ID=1000
ARG USER_GID=1000

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
 
# Add user.
RUN groupadd -g $USER_GID $USER \
    && useradd --uid $USER_ID --gid $USER_GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

RUN apt install sudo libgl1-mesa-glx mesa-utils libglapi-mesa libqt5gui5 -y
RUN apt-get install -y build-essential cmake git curl ca-certificates \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    wget \
    jupyter

RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/$USER/.local/bin"

RUN mkdir /.cache
COPY requirements.txt /.cache

USER $USER

# Install python requirements
RUN pip install -r /.cache/requirements.txt
RUN pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html



WORKDIR /home/$USER/Datasets
WORKDIR /home/$USER/workspace
ENTRYPOINT /bin/bash
