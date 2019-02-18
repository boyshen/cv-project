#!/usr/bin/python

import numpy as np
from Data_preprocessing import Data_Generator

def extract_feature(predict_dir,resnet_model,xception_model,inceptionV3_model):
    '''
    提取特征向量
    :param predict_dir: 图片目录
    :param resnet_model:  resnet 模型
    :param xception_model:  xception 模型
    :param inceptionV3_model: inceptionV3 模型
    :return: 特征向量
    '''

    resnet_gen = Data_Generator(predict_dir, (224, 224), 1, is_train=False)
    inception_gen = Data_Generator(predict_dir, (299, 299), 1, is_train=False)
    xception_gen = Data_Generator(predict_dir, (299, 299), 1, is_train=False)

    resnet_feature = resnet_model.predict_generator(resnet_gen,
                                                   len(resnet_gen),
                                                   verbose=0)

    xception_feature = xception_model.predict_generator(xception_gen,
                                                        len(resnet_gen),
                                                        verbose=0)

    inception_feature = inceptionV3_model.predict_generator(inception_gen,
                                                            len(resnet_gen),
                                                            verbose=0)

    feature = list()

    feature.append(np.array(resnet_feature))
    feature.append(np.array(inception_feature))
    feature.append(np.array(xception_feature))

    bottlebeck_features = np.concatenate(feature, axis=1)

    return bottlebeck_features

def predict_image(model, img_feature):
    '''
    预测图片
    :param model: 训练过的模型
    :param img_feature: 特征向量
    :return: 预测结果。字典
    '''

    predict_value = model.predict(img_feature, verbose=0)

    predict_value = predict_value.clip(min=0.005, max=0.995)

    result = {}

    if predict_value[0][0] >= 0.5:
        result['class'] = 'dog'
        result['pred'] = "{:.2f}%".format(predict_value[0][0] * 100)
    else:
        result['class'] = 'cat'
        result['pred'] = "{:.2f}%".format((1 - predict_value[0][0]) * 100)

    return result

if __name__ == '__main__':
    import config
    from keras.models import load_model
    from keras.applications.resnet50 import ResNet50
    from keras.applications.xception import Xception
    from keras.applications.inception_v3 import InceptionV3

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


    feature = extract_feature(config.img_path,
                              resnet_model,xception_model,inceptionV3_model)

    predcit_result = predict_image(model,feature)

    print(predcit_result)
