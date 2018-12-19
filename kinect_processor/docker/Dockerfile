FROM nvidia/cuda:9.0-cudnn7-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz sudo nano cmake && \
    apt-get install -y libavcodec-dev libjpeg-dev libavformat-dev libxvidcore-dev x264 libv4l-dev v4l-utils ffmpeg && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh 

   #wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    #echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    #/bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    #rm Miniconda3-4.2.12-Linux-x86_64.sh

RUN apt-get update && \
     apt-get install -y libglew1.5 libglew-dev libglfw3-dev

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
#    apt-get install -y patchelf && \
    apt-get install unzip && \
    pip install --upgrade pip && \
#    pip install tensorflow-gpu pydot-ng keras && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
#    conda install theano pygpu && \
#    pip install keras && \
    pip install opencv-python && \
    pip install imgaug && \
#    pip install torch torchvision && \
    pip install chainer && \
    pip install cupy-cuda90 && \
    conda clean -yt

ENV NB_USER yordan
ENV NB_UID 1000
    
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    echo "yordan:yordan" | chpasswd && \
    adduser yordan sudo && \
    mkdir -p $CONDA_DIR && \
    chown yordan $CONDA_DIR -R && \
    mkdir -p /src && \
    chown yordan /src

USER yordan
WORKDIR /home/yordan

ENV PYTHONPATH='/src/:$PYTHONPATH'

#RUN mkdir -p .mujoco \
#    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
#    && unzip mujoco.zip -d .mujoco \
#    && rm mujoco.zip
#COPY ./mjkey.txt .mujoco/

#RUN git clone https://github.com/openai/mujoco-py.git mujoco-py && \
#    cd mujoco-py && \
#    pip install -r requirements.txt && \
#    pip install -r requirements.dev.txt && \
#    LD_LIBRARY_PATH=$/.mujoco/mjpro150/bin pip install -e .
    
#RUN git clone https://github.com/openai/gym gym && \
#    cd gym && \
#    pip install -e .

#RUN pip install mujoco_py==0.5.7


RUN pip install --upgrade cython && \
    pip install fcn


#ENV LD_LIBRARY_PATH /home/yordan/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# we need this because https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# set aliases
RUN echo 'alias notebook="jupyter notebook --ip=0.0.0.0"' >> ~/.bashrc
RUN echo 'alias cl="clear"' >> ~/.bashrc

# tmux config
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

RUN pip install ipympl
