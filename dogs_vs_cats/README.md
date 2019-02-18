## 项目概述

项目来自kaggle的猫狗大战。主要实现输入一张彩色图片，识别彩色图片中的图像，是猫、还是狗。

项目使用flask网页应用框架，实现在web端上传一张图片，识别图像是猫，还是狗。

kaggle项目地址：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

## 机器硬件

采用 aws 云服务器 p3.2xargs 实例。

## 系统环境

ubuntu 操作系统。在aws配置系统环境中搜 udacity-aind2 

## 训练时间

整个训练时间不超过5个小时

## 相关库

主要使用的库:

numpy=1.15.4
keras=2.2.4
matplotlib=3.0.2
tensorflow=1.12.0
h5py=2.9.0
pandas=0.24.0

详细可见目录中文件：requirements/dogs_vs_cats_linux-pip.txt

## 项目结构

.
├── abnormal_picture		# 项目中检查的异常图片信息
│   ├── abnormal_image.txt  					
│   ├── resnet50_check.txt
│   └── xception_check.txt
├── bottlebeck_features     # 使用resnet50、inceptionV3、xception 提取的特征向量文件
│   ├── InceptionV3_bottlebeck_features.h5py
│   ├── resnet50_bottlebeck_features.h5py
│   └── xception_bottlebeck_features.h5py
├── Data_preprocessing.py   # 图片处理python文件。包含图片处理的相关源码
├── extract_bottleneck_features.py # 提取特征向量的python文件
├── images  # 项目文件中的图片
│   ├── CNN.png
│   ├── data-dir.png
│   ├── datadir.png
│   ├── Imagenet.png
│   ├── kaggle_score.png
│   ├── kaggle-score.png
│   ├── keras_modle.png
│   ├── model.png
│   ├── Neuron2.png
│   ├── Neuron3.png
│   ├── ResNet0.png
│   ├── ResNet2.png
│   ├── ResNet3.png
│   ├── symlink1.png
│   ├── webapp-dir.png
│   └── web_predict.gif
├── model  # 训练好的最佳模型
│   └── dogs_vs_cats_best_model.hdf5
├── model_func.py  # 模型python文件，包含构建模型、训练模型等
├── predict_dogs_vs_cats.csv  # 预测结果
├── __pycache__
│   ├── Custom_model.cpython-36.pyc
│   ├── Data_preprocessing.cpython-36.pyc
│   ├── model_func.cpython-36.pyc
│   ├── Resnet50_preprocessing.cpython-36.pyc
│   └── Xception_preprocessing.cpython-36.pyc
├── README.md
├── requirements  # 环境配置文件
│   ├── dogs_vs_cats_linux-pip.txt
│   └── dogs_vs_cats_linux.yaml
├── Resnet50_preprocessing.py   # 使用resnet50检测异常图片的python文件
├── Untitled.ipynb
├── Web_APP      # web 应用程序目录
│   ├── config.py    # web 应用配置文件
│   ├── Data_preprocessing.py
│   ├── dogs_vs_cats_linux-pip.txt   # 环境依赖库
│   ├── dogs_vs_cats_linux.yaml      # conda 构建环境文件
│   ├── image_predict.py
│   ├── __pycache__
│   │   ├── config.cpython-36.pyc
│   │   ├── Data_preprocessing.cpython-36.pyc
│   │   └── image_predict.cpython-36.pyc
│   ├── static
│   │   ├── css
│   │   │   └── index.css
│   │   ├── images
│   │   │   ├── loading.gif
│   │   │   └── woof_meow.jpg
│   │   └── js
│   │       └── jquery-1.9.1.min.js
│   ├── templates
│   │   ├── index.html
│   │   └── woof_meow.jpg
│   └── web_app.py    # web应用程序启动文件。使用python web_app.py 启动文件。
└── Xception_preprocessing.py # 使用xception 检查异常数据python文件

13 directories, 54 files

特征向量文件如果需要可在此下载：

链接:https://pan.baidu.com/s/1NUzNzdVmZEj8ImCK5zx_1w  密码:54ap

## 项目步骤

1. 安装相关库

	```bash
	conda env create -f requirements/dogs_vs_cats_linux.yaml
	source activate dogs_vs_cats
	pip install -r dogs_vs_cats_linux-pip.txt
	```
2. 打开 notebook

```
jupyter notebook Untitled.ipynb
```

3. 启动网页应用

```
cd Web_APP
python web_app.py
```
注意查看 config.py 配置文件。配置文件中可修改监听的网络地址和端口，以及上传图片的存储路径，预训练模型文件路径

![title](images/web_predict.gif)



