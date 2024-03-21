import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
from scipy import stats
import h5py
import random

def generator(passes=np.inf,batchSize=1,num_data_train = 256,path= r'./grutraindata.h5',txtpath=r'./grutraindata.txt'):
    h5 = h5py.File(path, "r", libver="latest", swmr=True)
    epochs=0
    file = open(txtpath, 'r')
    X = file.readlines() 
    n=len(X)
    for i in range(n):
        X[i]=X[i][:-1]
    data_list =X  
    while epochs<passes:

        for i in range(0,num_data_train,batchSize):
            tmpx = []
            tmpy = []
            for m in data_list[i:i+batchSize]:
                data = h5[m][()]
                event = data[:,1:5]
                labelt = 1-data[:,-1:]
                labelf = data[:,-1:]
                label= np.concatenate((labelt,labelf),axis=1)
                tmpx.append(event)
                tmpy.append(label)
            datax= np.array(tmpx)
            datay = np.array(tmpy)
            yield datax, datay

def validgenerator(passes=np.inf,batchSize=128,ini_num_valid = 128,path= r'./grutraindata.h5',txtpath=r'./grutraindata.txt',num_data_valid=5120):
    h5 = h5py.File(path, "r", libver="latest", swmr=True)
    epochs=0
    file = open(txtpath, 'r')
    X = file.readlines() 
    n=len(X)
    for i in range(n):
        X[i]=X[i][:-1]
    data_list =X   
    while epochs<passes:
        for i in range(ini_num_valid,ini_num_valid+num_data_valid,batchSize):
            tmpx = []
            tmpy = []
            for m in data_list[i:i+batchSize]:
                data = h5[m][()]
                event = data[:,1:5]
                labelt = 1-data[:,-1:]
                labelf = data[:,-1:]
                label= np.concatenate((labelt,labelf),axis=1)
                tmpx.append(event)
                tmpy.append(label)
            datax= np.array(tmpx)
            datay = np.array(tmpy)
            yield datax, datay

if __name__ == "__main__":
    import sys
    sys.path.append('./su')
    from GRU import GRU_PA 
    model = GRU_PA()
    model.summary()

    model.layers
    model.summary()
    num_data_train = 3240000-324000
    ini_num_valid = 3240000-324000
    num_data_valid=324000
    save_dir=r'./grutrain'
    checkpointer = ModelCheckpoint(os.path.join(save_dir, 'gru_{epoch:02d}_{val_loss:.2f}.hdf5'),
                                   verbose=0, save_weights_only=True)
                              
    model.fit_generator(generator(passes=np.inf,
                                  batchSize=128,
                                  num_data_train = num_data_train ,
                                  path=r'./grutraindata.h5',txtpath='./grutraindata.txt'),
                        steps_per_epoch=num_data_train/128,
                        validation_data=validgenerator(passes=np.inf,
                                                       batchSize=128 ,
                                                       ini_num_valid=ini_num_valid,
                                                       path=r'./grutraindata.h5',txtpath='./grutraindata.txt',
                                                       num_data_valid=num_data_valid),
                        validation_steps=num_data_valid/128,
                        epochs=10, verbose=1, callbacks=[checkpointer])
