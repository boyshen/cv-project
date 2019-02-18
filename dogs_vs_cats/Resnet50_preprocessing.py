#!/usr/bin/python

import numpy as np
import Data_preprocessing
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions

from keras.preprocessing import image

class Resnet50_preprocess(object):

    def __init__(self):
        self.Resnet50_model = ResNet50(weights='imagenet')

    def model_predict(self,top_s,img_path):
        img = preprocess_input(Data_preprocessing.path_to_tensor(img_path))
        pred = self.Resnet50_model.predict(img)
        decodes,_,_= zip(*decode_predictions(pred,top=top_s)[0])
        return decodes

if __name__ == '__main__':
    model = Resnet50_preprocess()
    decode = model.model_predict(top_s=3,img_path='data/train/cat.773.jpg')
    print(decode)