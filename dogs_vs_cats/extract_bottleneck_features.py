#!/usr/bin/python

import numpy as np
from keras.preprocessing import image

def path_to_tensor(img_path,target_size=(224,224)):
    """
    将一张图片转换为张量
    :param img_path: 图片的绝对路径
    :return: 图片张量
    """
    img = image.load_img(img_path,target_size=target_size)  #加载图片
    x = image.img_to_array(img)     #将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    return np.expand_dims(x,axis=0) #将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回

def paths_to_tensor(img_paths,target_size):
    """
    将多张图片转换为张量
    :param img_paths: 多张图片的路径
    :return: 图片张量
    """
    list_of_tensor = [path_to_tensor(img,target_size) for img in img_paths]
    return np.vstack(list_of_tensor)

def extract_resnet50(model,tensor):
    from keras.applications.resnet50 import preprocess_input
    return model.predict(preprocess_input(tensor))

def extract_xception(model,tensor):
    from keras.applications.xception import preprocess_input
    return model.predict(preprocess_input(tensor))

def extract_InceptionV3(model,tensor):
    from keras.applications.inception_v3 import preprocess_input
    return model.predict(preprocess_input(tensor))

def extract_model_feature(resnet50_model,xception_model,inceptionV3_model,
                          image):
    tensor = path_to_tensor(image,target_size=(224,224)).astype('float32')/255
    resnet_feature = np.array(extract_resnet50(resnet50_model,tensor))

    tensor = path_to_tensor(image,target_size=(299,299)).astype('float32')/255
    xception_feature = np.array(extract_xception(xception_model,tensor))

    inceptionV3_feature = np.array(extract_InceptionV3(inceptionV3_model,tensor))

    #print(resnet_feature.shape)
    #print(xception_feature.shape)
    #print(inceptionV3_feature.shape)

    return np.concatenate([resnet_feature,inceptionV3_feature,xception_feature],axis=1)


