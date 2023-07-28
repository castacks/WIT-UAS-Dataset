# CUDA version: 11.8, multiple clamped dependencies, DON'T EVER TOUCH THIS UNLESS YOU KNOW WHAT YOU ARE DOING!!!
# e.g. CUDA 11.8, cuDNN => VPI (depends on CUDA, to be installed), PyTorch 1.13.0, TensorRT => Torch-TensorRT 1.3.0
# pytorch:22.xx-py3 note 22.xx <= 22.10 uses conda for pytorch and torch_tensorrt, and sets the conda python as default python, which is problematic for ros tools
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.06-py3
ENV HOME_FOLDER=/root
WORKDIR $HOME_FOLDER/

# Fix apt install stuck problem
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# update all obsolete packages to latest, install sudo, and cleanup
RUN apt update -o Acquire::Check-Valid-Until=false -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt full-upgrade -y && \
    apt install -y sudo && \
    apt autoremove -y && \
    apt autoclean -y

# fix local time problem
RUN apt install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

ENV PYTHON_VERSION=3.10

# # install python3 and pip3
# RUN apt install -y --no-install-recommends python3-dev python3-pip python3-setuptools python3-wheel && \
#     pip3 install --upgrade pip

# set default python version to 3 (${PYTHON_VERSION})
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 2 && \
    update-alternatives --set python /usr/bin/python3

# # set default pip version to pip3
# RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 2 && \
#     update-alternatives --set pip /usr/local/bin/pip3

# install some goodies
RUN apt install -y --no-install-recommends lsb-release apt-utils software-properties-common zsh unzip ncdu git less screen tmux tmuxp tree locate perl net-tools vim nano emacs htop curl wget build-essential ffmpeg

# # Add a new group and user
# RUN addgroup --gid 1000 $USER && \
#     adduser --uid 1000 --ingroup $USER --home $HOME_FOLDER --shell /bin/zsh --disabled-password --gecos "" $USER && \
#     echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # Fix UID/GID when mounting from host using this: https://github.com/boxboat/fixuid
# RUN curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
#     chown root:root /usr/local/bin/fixuid && \
#     chmod 4755 /usr/local/bin/fixuid && \
#     mkdir -p /etc/fixuid

# # Switch to the new user
# USER $USER:$USER

# copy oh-my-zsh powerlevel10k theme config
COPY --from=home-folder-config ./.p10k.zsh $HOME_FOLDER/

# install zsh, Oh-My-Zsh, and plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/latest/download/zsh-in-docker.sh)" -- \
    -t https://github.com/romkatv/powerlevel10k \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p https://github.com/CraigCarey/gstreamer-tab \
    -a "[[ ! -f $HOME_FOLDER/.p10k.zsh ]] || source $HOME_FOLDER/.p10k.zsh" \
    -a "bindkey -M emacs '^[[3;5~' kill-word" \
    -a "bindkey '^H' backward-kill-word" \
    -a "autoload -U compinit && compinit" \
    -a "export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH}"

# Append libtorch, TensorRT, torch_tensorrt library path to LD_LIBRARY_PATH
#! /opt/hpcx/ompi/lib is to fix python3 import torch bug: OSError: /opt/hpcx/ompi/lib/libmpi.so.40: undefined symbol: opal_hwloc201_hwloc_get_type_depth
RUN echo "export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/lib:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt:/usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch_tensorrt/lib:${LD_LIBRARY_PATH}" >> $HOME_FOLDER/.bashrc

# change default shell for the $USER in the image building process for extra environment safety
RUN chsh -s $(which zsh)

# end of sudo apt installs
RUN apt full-upgrade -y && \
    apt autoremove -y && \
    apt autoclean -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*
# Patch bug introduced by libtorch: not prepending namespace c10:: to nullopt and optional< > in some files
RUN perl -pi -e 's/(?<!c10::)optional</c10::optional</g; s/(?<!c10::)nullopt/c10::nullopt/g' /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/include/ATen/ops/from_blob.h /usr/local/lib/python${PYTHON_VERSION}/dist-packages/torch/include/ATen/DeviceGuard.h

# # Copy fixuid config
# COPY --from=home-folder-config ./fixuid-config.yml /etc/fixuid/config.yml

# Copy screenrc and tmux config to enable mouse scrolling
COPY --from=home-folder-config ./.screenrc $HOME_FOLDER/
COPY --from=home-folder-config ./.tmux.conf $HOME_FOLDER/

# albumentations, wandb, SAHI, and YOLOs
RUN pip3 install albumentations terminaltables torchsummary wandb sahi yolov5 yolov8

# Install pre-commit for convenience
RUN pip3 install pre-commit

# git set safe directory
RUN git config --global --add safe.directory "*"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Set the default shell to zsh
SHELL [ "/bin/zsh", "-c" ]

# Entrypoint command
# ENTRYPOINT [ "/bin/sh" , "-c", "fixuid; /bin/zsh" ]
ENTRYPOINT [ "/opt/nvidia/nvidia_entrypoint.sh", "/bin/zsh" ]
