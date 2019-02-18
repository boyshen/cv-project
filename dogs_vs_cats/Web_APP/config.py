#!/usr/bin/python
# 创建日期：2019.02.01
# 发布日期: 10/02/2019
# version: 1.0
# PURPOSE：web应用配置文件

# 是否开启调试模式
debug = False

# 监听网络IP地址
host = '0.0.0.0'
port = 8888

# 上传图片保存目录, 注意最后目录不要有‘/’
img_path = '/data/dogs_vs_cats/data/predict'

# 模型权重
model_weight = '/data/dogs_vs_cats/model/dogs_vs_cats_best_model.hdf5'

# 允许上传的图片类型
ALLOWED_EXTENSIONS = ['png', 'jpg', 'JPG', 'PNG']

