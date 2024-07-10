# Transformers开发环境搭建
## 步骤
- 系统准备
- Miniconda
- 拉取代码
- Jupyter Lab

## 系统准备

### 创建新用户
```shell
adduser wood
usermod -aG sudo wood
```
logout切换到新用户

### 安装 [CUDA 12.04](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)（包含GPU驱动）：

```shell
$ wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
$ sudo sh cuda_12.2.2_535.104.05_linux.run

$ vi ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin  
export LD_LIBRARY_PATH=/usr/local/cuda/lib64  

# 查看GPU驱动版本
$ nvidia-smi

# 查看CUDA版本
$ nvcc --version
```

### 安装ffmpeg

```shell
#sudo apt update && sudo apt upgrade
sudo apt install ffmpeg
ffmpeg -version
```


## Miniconda

### 安装Miniconda

```shell
$ mkdir -p ~/miniconda3
$ wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh

# 验证Miniconda是否安装成功
$ conda --version
```

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
$ conda create -n transformers python=3.11
$ conda activate transformers
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
$ git clone https://github.com/blueofsky/LLM-quickstart.git
```
### 安装依赖
```shell
$ pip install -r requirements.txt
```

## Jupyter Lab
### 安装Jupyter Lab

```shell
$ conda install jupyterlab
```

### 生成配置文件

```shell
$ jupyter lab --generate-config
```

### 修改`jupyter_lab_config.py`

```python
c.ServerApp.allow_root = True # 非 root 用户启动，无需修改
c.ServerApp.ip = '*'
```

### 后台启动
```shell
$ nohup jupyter lab --port=8800 --NotebookApp.token='666666' --notebook-dir=./ &
```

