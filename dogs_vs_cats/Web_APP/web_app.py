#!/usr/bin/python
# 创建日期：2019.02.01
# 发布日期: 10/02/2019
# version: 1.0
# PURPOSE：kaggle项目-猫狗大战项目web应用程序
#
# 应用说明：
#   访问web页面，在web页面上选择上传一张猫或狗的图片，程序通过加载预先训练的卷积神经网络模型， 识别上传的图片类别（即猫或狗）。
#
# 神经网络模型访问：
#
# kaggle 项目访问i地址：
#   https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
#
# 服务启动：
#   python web_app.py
#
# 服务配置：
#   当前目录下config.py 文件


import os
import config
import time
import tensorflow as tf
from flask import Flask
from flask import render_template,request,redirect,url_for,make_response,jsonify
from werkzeug import secure_filename
from image_predict import extract_feature,predict_image

from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload',methods=['POST'])
def upload():
    print("upload file ...")

    file = request.files['file']
    file_type = str(secure_filename(file.filename)).split('.')[-1]

    if file_type in config.ALLOWED_EXTENSIONS:

        # 以时间戳命名目录名
        dir_name = str(round(time.time() * 1000))
        # 以时间戳命名文件，防止文件名相同
        name =  dir_name + '.' + file_type
        print("rename filename {} ".format(name))

        # 创建目录
        img_path = config.img_path + '/' + dir_name + '/' + 'predict'
        os.makedirs(img_path)

        file_path = os.path.join(img_path,name)
        file.save(file_path)

        print("upload file {}".format(secure_filename(file.filename)))

        print("start predict image ...")
        predcit_result = image_del(dir_name)

        return jsonify({"status": 0,
                        'file': name,
                        'class':predcit_result['class'],
                        'pred':predcit_result['pred']})
    else:
        return jsonify({"status": -1, 'msg':'Error ! 不支持该文件类型 !'})

@app.route('/show_image/<string:filename>',methods=['GET'])
def show_img(filename):

    print("show image ...")

    if request.method == 'GET':
        if filename is None:
            pass
        else:
            # 根据文件名找到文件所在路径
            time_name  = str(filename).split('.')[0]
            img_path = config.img_path + '/' + time_name + '/' + 'predict'

            file_dir = os.path.join(img_path,str(filename))
            img_data = open(file_dir,'rb').read()
            response = make_response(img_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

def image_del(dir_name):

    with graph.as_default():
        # 获取图片特征
        feature = extract_feature(config.img_path + '/' + dir_name,
                              resnet_model, xception_model, inceptionV3_model)

        # 预测图片
        predcit_result = predict_image(model, feature)

    return predcit_result

def check_config():

    # 检测上传图片的路径是否存在
    if not os.path.exists(config.img_path):
        os.makedirs(config.img_path)

    if not os.path.exists(config.model_weight):
        print("model weight not found ! {}".format(config.model_weight))
        return False

def init_model():
    # 全局化模型
    global resnet_model
    global xception_model
    global inceptionV3_model
    global model

    # 初始化 resnet50 模型
    resnet_model = ResNet50(input_shape=(224, 224, 3),
                            weights='imagenet',
                            include_top=False,
                            pooling='avg')

    # 初始化 xception 模型
    xception_model = Xception(input_shape=(299, 299, 3),
                              weights='imagenet',
                              include_top=False,
                              pooling='avg')

    # 初始化 inceptionV3 模型
    inceptionV3_model = InceptionV3(input_shape=(299, 299, 3),
                                    weights='imagenet',
                                    include_top=False,
                                    pooling='avg')

    model = load_model(config.model_weight)

    global graph
    graph = tf.get_default_graph()

    #return resnet_model,xception_model,inceptionV3_model,model

def init():

    # 检查配置
    if check_config() is False:
        return False

def main():

    # 初始化服务
    init()

    # 初始化模型
    init_model()

    # 启动服务
    app.run(host=config.host, debug=config.debug, port=config.port)

if __name__ == '__main__':
    main()
    #feature = extract_feature(config.img_path + '/' + '1550317822380',
    #                          resnet_model, xception_model, inceptionV3_model)

    # 预测图片
    #predcit_result = predict_image(model, feature)
    #print(predcit_result)