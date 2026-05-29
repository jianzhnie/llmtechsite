# HuggingFace 模型上传指南

在当今的人工智能领域，模型的共享与交流变得愈发重要，而 HuggingFace 作为一个知名的模型托管平台，为开发者们提供了便捷的模型展示和交流空间。本文将详细介绍如何将本地训练好的模型上传至 HuggingFace，涵盖从安装必要工具、登录平台、创建项目，到最终成功提交模型的全过程。

## 1. 安装 git-lfs 和 huggingface_hub

通过CLI上传 Hugging Face同样是跟Git相关联，通常大模型的模型文件都比较大，因此我们需要安装git lfs，对大文件系统支持

```shell
# Linux
apt-get install git-lfs
# Macos
brew instll git-lfs
git lfs install
pip install huggingface_hub
```
## 2. huggingface-cli login命令登录

使用 huggingface-cli login命令进行登录，登录过程中需要输入用户的Access Tokens，获取时，需要先验证email

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
## 5. 通过git提交到远程仓库

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
