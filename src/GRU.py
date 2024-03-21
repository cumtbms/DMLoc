import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
# from keras.layers import LSTM,Dropout,Dense,normalization,Conv1D,MaxPool1D,Dense,Conv2D,MaxPool2D,Input, merge, UpSampling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
import keras.backend as K
import keras

def GRU_PA():
    inputs=Input((None,4))
    gru1 = Bidirectional(GRU(16, return_sequences=True))(inputs)
    gru2 = Bidirectional(GRU(16, return_sequences=True))(gru1)
    dense=Dense(2,activation='softmax')(gru2)
    model = Model(inputs=inputs, outputs=dense)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # model_2d_segunet_decoderd_lrelu import *#model_2d_segunet_decoderd_lrelu_01 import *#model_add_merge_bn import *# model_segunet_dd_lrelu01_dropout
    # len = 2888
    # input_shape = (len,  3)
    model = GRU_PA()
    model.summary()