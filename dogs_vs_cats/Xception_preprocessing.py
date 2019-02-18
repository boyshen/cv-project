#!/usr/bin/python

import numpy as np
import Data_preprocessing
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input,decode_predictions

from keras.preprocessing import image

class Xception_preprocess(object):

    def __init__(self):
        self.xception_model = Xception(weights='imagenet')

    def model_predict(self,top_s,img_path):
        img = preprocess_input(Data_preprocessing.path_to_tensor(img_path))
        pred = self.xception_model.predict(img)
        decodes,_,_= zip(*decode_predictions(pred,top=top_s)[0])
        return decodes

if __name__ == '__main__':
    model = Xception_preprocess()
    decode = model.model_predict(top_s=3,img_path='data/train/cat.773.jpg')
    print(decode)