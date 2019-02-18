#!/usr/bin/python

import h5py
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.engine.input_layer import Input
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
#from keras.applications.resnet50 import ResNet50

def save_bottlebeck_features(input_shape, train_gen, test_gen, application_model, train_dataset_name='train', 
                             test_dataset_name='test',label_name='label',file_name='bottlebeck_features'):
    """
    使用预训练模型提取瓶颈特征并保存
    :param input_shape: tuple，图片尺寸
    :param train_gen: 训练数据生成器
    :param test_gen: 测试数据生成器
    :param application_model: 模型对象
    :param dataset_name: 数据集名称
    :param label_name: 数据集标签名
    :param file_name: 保存的文件名
    :return: 
    """
    
    if os.path.exists(file_name):
        print("{} already exist! Please remove or rename".format(file_name))
        return False
    
    model = application_model(input_shape=input_shape,
                              weights='imagenet',
                              include_top=False,
                              pooling='avg')

    X_train = model.predict_generator(train_gen, len(train_gen), verbose = 1)
    X_test = model.predict_generator(test_gen, len(test_gen), verbose = 1)
    #data = model.predict_generator(generator, 100, verbose = 1)

    with h5py.File(file_name) as h:
        h.create_dataset(train_dataset_name, data=X_train)
        h.create_dataset(test_dataset_name, data=X_test)
        h.create_dataset(label_name, data=train_gen.classes)
            
    print("save bottlebeck features : {}".format(file_name))
    
def get_bottlebeck_features(features_file,train_dataset_name='train',test_dataset_name='test',label_name='label'):
    '''
    根据文件，提取特征数据集
    :param features_file: list，瓶颈特征文件路径 
    :return: 训练、测试、标签数据集
    '''
    np.random.seed(47)
    
    X_train = []
    X_test = []
    y_train = None
    
    for filename in features_file:
        print("load %s ..." %filename)
        with h5py.File(filename,'r') as h:
            X_train.append(np.array(h[train_dataset_name]))
            X_test.append(np.array(h[test_dataset_name]))
            y_train = np.array(h[label_name])
                
    X_train = np.concatenate(X_train,axis=1)
    X_test = np.concatenate(X_test,axis=1)
    
    X_train,y_train = shuffle(X_train,y_train)
    
    return X_train,X_test,y_train

def create_model(train_data,drop_out=0.5,optimizer='rmsprop'):
    """
    构建模型
    :param train_data: 提取特征的数据集 
    :param drop_out: 
    :param optimzier: 优化器
    :return: 
    """
    
    input_layer = Input(shape=train_data.shape[1:])
    model = input_layer
    model = Dropout(drop_out)(model)
    #model = Dense(1024,activation='relu')(model)
    #model = Dropout(drop_out)(model)
    model = Dense(512,activation='relu')(model)
    model = Dropout(drop_out)(model)
    #model = Dense(256,activation='relu')(model)
    #model = Dropout(drop_out)(model)
    #model = Dense(128,activation='relu')(model)
    #model = Dropout(drop_out)(model)
    #model = Dense(64,activation='relu')(model)
    #model = Dropout(drop_out)(model)
    model = Dense(1, activation='sigmoid')(model)
    model = Model(input_layer,model)

    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model,X_train,y_train,check_point,
                batch_size=64,n_epochs=10,verbose=1,valid_split=0.2):
    """
    训练模型
    :param model: 模型对象 
    :param X_train: 训练数据集
    :param y_train: 验证数据集
    :param check_point: 检查点文件路径
    :param batch_size: 每个梯度更新的样本数
    :param n_epochs: 轮数
    :param verbose: 输出显示
    :param valid_split: 用作验证数据的训练数据的分数
    :return: 
    """

    check_point = ModelCheckpoint(filepath=check_point,
                                  verbose=verbose,save_best_only=True)

    history = model.fit(X_train,y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=verbose,
                        callbacks=[check_point],
                        validation_split=valid_split)

    return history

def plot_train_accuracy_and_loss(history,figsize=(12,6)):
    """
    绘制训练的准确率和损失曲线
    :param history: history 对象，训练模型时的历史记录
    :param figsize: tuple，画布尺寸
    :return:
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=figsize)

    # 绘制准确率曲线
    plt.subplot(1,2,1)
    plt.plot(epochs,acc,'r-',label='accuracy')
    plt.plot(epochs,val_acc,'g-',label='validation accuracy')
    plt.ylabel("accuray")
    plt.xlabel("epochs")
    plt.title("Accuracy of train and validation")
    plt.legend()

    # 绘制损失曲线
    plt.subplot(1,2,2)
    plt.plot(epochs,loss,'r-',label='loss')
    plt.plot(epochs,val_loss,'g-',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title("Loss of train and validation")
    plt.legend()

    plt.show()
    
def save_csv(y_pred,test_gen,csv_file,pred_file='predict.csv'):
    """
    对预测结果按照要求保存为csv文件
    :param y_pred: 预测结果
    :param test_gen: test数据集生成器
    :param csv_file: 提供的csv文件模版
    :param pred_file: 输出保存的预测文件
    :return: 
    """

    df = pd.read_csv(csv_file)

    for i,file_name in enumerate(test_gen.filenames):
        index = int(file_name.split('/')[1].split('.')[0]) - 1
        df.set_value(index,'label',y_pred[i])

    df.to_csv(pred_file,index=None)
    print("save csv file : {}".format(pred_file)) 
  


