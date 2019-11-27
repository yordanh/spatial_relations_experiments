FROM nvidia/cuda:8.0-cudnn7-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz sudo nano cmake && \
    apt-get install -y libavcodec-dev libjpeg-dev libavformat-dev libxvidcore-dev x264 libv4l-dev v4l-utils ffmpeg && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

# Python
ARG python_version=2.7.11

###
# OpenCV (and skvideo for recording)
# Autograd, Chainer and PyMC3
RUN conda install -y python=${python_version} && \
    apt-get install -y htop && \
    apt-get update --fix-missing && \
    apt-get install -y imagemagick && \
    apt-get install -y tmux && \
    pip install --upgrade pip && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    pip install opencv-python && \
    pip install imgaug && \
    pip install chainer && \
    pip install cupy-cuda80 && \
    pip install ipympl && \
    conda clean -yt

ENV NB_USER user
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    echo $NB_USER:$NB_USER | chpasswd && \
    adduser $NB_USER sudo && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER user

ENV PYTHONPATH='/src/:$PYTHONPATH'

######################################################
# tmux bindings and aliases follow; could be ignored #
######################################################

RUN echo 'alias notebook="jupyter notebook --ip=0.0.0.0"' >> ~/.bashrc
RUN echo 'alias cl="clear"' >> ~/.bashrc

RUN echo 'unbind C-b' >> ~/.tmux.conf
RUN echo 'set-option -g prefix C-a' >> ~/.tmux.conf
RUN echo 'bind-key C-a send-prefix' >> ~/.tmux.conf

RUN echo 'bind _ split-window -h' >> ~/.tmux.conf
RUN echo 'bind - split-window -v' >> ~/.tmux.conf
RUN echo "unbind '\"'" >> ~/.tmux.conf
RUN echo 'unbind %' >> ~/.tmux.conf

RUN echo 'bind -n M-Left select-pane -L' >> ~/.tmux.conf
RUN echo 'bind -n M-Right select-pane -R' >> ~/.tmux.conf
RUN echo 'bind -n M-Up select-pane -U' >> ~/.tmux.conf
RUN echo 'bind -n M-Down select-pane -D' >> ~/.tmux.conf

RUN echo 'set-window-option -g xterm-keys on' >> ~/.tmux.conf