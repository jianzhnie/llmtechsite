# HuggingFace 模型上传指南

在当今的人工智能领域，模型的共享与交流变得愈发重要。HuggingFace 作为知名的模型托管平台，为开发者提供了便捷的模型展示和交流空间。本文将详细介绍如何将本地训练好的模型上传至 HuggingFace，涵盖从安装必要工具、登录平台、创建项目到最终成功提交模型的全过程。

## 1. 安装 git-lfs 和 huggingface_hub

通过 CLI 上传 HuggingFace 同样与 Git 相关联。通常大模型的模型文件都比较大，因此我们需要安装 Git LFS 以支持大文件系统。

```shell
# Linux
apt-get install git-lfs
# Macos
brew install git-lfs
git lfs install
pip install huggingface_hub
```
## 2. huggingface-cli login命令登录

使用 `huggingface-cli login` 命令进行登录。登录过程中需要输入用户的 Access Token，获取时需要先验证邮箱。

```shell
git config --global credential.helper store
huggingface-cli login
```
## 3. 创建项目

```shell
huggingface-cli repo create PCL-Reasoner/V1
git clone https://hf-mirror.com/PCL-Reasoner/V1
```
## 4. 把训练好的模型保存进里面

```shell
cp -r hf_model_ckpt/* V1
```
## 5. 通过 Git 提交到远程仓库

```shell
cd V1
git add .
git commit -m "add:PCL-Reasoner/V1"
git push
```

输入官方推荐命令

```shell
huggingface-cli upload PCL-Reasoner/V1 .
```
