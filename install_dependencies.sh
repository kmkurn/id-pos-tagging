#!/bin/sh

# Install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p "$HOME/miniconda" && \
    echo "export PATH=\"$HOME/miniconda/bin:$PATH\"" >> ~/.bashrc && \
    # Install packages
    export PATH="$HOME/miniconda/bin:$PATH" && \
    conda update -y conda && conda env create -f environment.yml
