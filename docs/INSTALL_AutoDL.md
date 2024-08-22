# Transformers开发环境搭建
## 步骤
- 系统准备
- Miniconda
- 拉取代码
- Jupyter Lab

## 系统准备

### 安装 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)（包含GPU驱动）：

```shell
wget wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sh cuda_12.2.2_535.104.05_linux.run

```
### 环境配置 ~/.bashrc
```shell
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/
```
### 重新登录，查看CUDA版本
```shell
$ nvidia-smi
$ nvcc --version
```

## Miniconda

### 配置Miniconda

```shell
$ vi ~/.condarc
auto_activate_base: False
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
show_channel_urls: True
```

### 创建虚拟环境
```shell
mkdir -p /root/autodl-tmp/conda/pkgs
mkdir -p /root/autodl-tmp/conda/envs
conda config --add pkgs_dirs /root/autodl-tmp/conda/pkgs
conda config --add envs_dirs /root/autodl-tmp/conda/envs

conda create -n transformers python=3.11
conda init bash && source /root/.bashrc
conda activate transformers
```

## 拉取代码

### 配置git

```shell
$ vi ~/.gitconfig
[user]
	name = blueofsky
    email = myjobmails@163.com    
[alias]
	co = checkout
	ci = commit
	br = branch
	st = status
[push]
    default = simple
[color]
    ui = auto
```

### 下载
```shell
git clone https://github.com/blueofsky/LLM-quickstart.git
```
### 安装依赖
```shell
pip install -r requirements.txt
```

## Jupyter Lab
### 安装Jupyter Lab

```shell
conda install ipykernel
ipython kernel install --user --name=transformers
```



