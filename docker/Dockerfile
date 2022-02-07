FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04


ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Kiev
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    zip \
    x11vnc \
    xvfb \
    psmisc \
    ca-certificates \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# Install python
ENV PYTHON_VERSION=3.8
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH


# Create user
ARG UID=1001
ARG USER_NAME="dev"
RUN useradd -u $UID -m $USER_NAME && \
    echo "${USER_NAME}: " | chpasswd && adduser $USER_NAME sudo && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME
USER $USER_NAME
WORKDIR /home/$USER_NAME


# Install ViZDOOM
RUN sudo apt-get update && sudo apt-get install -y build-essential zlib1g-dev libsdl2-dev \
    nasm tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev \
    libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev julia
RUN pip install numpy
RUN pip install vizdoom
RUN pip install pynput
RUN pip install fire
RUN pip install tqdm


# Install PyTorch
RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install pytorch-lightning==1.4.9
RUN pip install efficientnet-pytorch==0.7.1
RUN pip install opencv-python-headless
RUN pip install scikit-image


# Install dotfiles
RUN git clone https://github.com/IlliaOvcharenko/dotfiles
RUN cd dotfiles && ./install && cd ../
RUN echo dprompt >> /home/$USER_NAME/.bashrc


ENV DISPLAY=:1
CMD ["/bin/bash"]

