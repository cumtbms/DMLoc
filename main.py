import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.models import *
from keras.layers import LSTM, Dropout, Dense, normalization, Conv1D, MaxPool1D, Dense, Conv2D, MaxPool2D
from keras import backend as K
import os
import sys, string, shutil
import pickle
import math
import os
from keras.backend import clear_session
from src.apsemenergy import sembalance_ap_PSGPU
from src.GRU import GRU_PA
from src.peak import getres
from numba import cuda
import scipy.signal as sgn
from src.model_2d_segunet_decoderd_lrelu_01 import *



def pickpoint(datap, datas, samplerate=1000, minDistance=40, pWindowLen=31, sWindowLen=31, tracenum=69, wintime=6,
              height=0.15, reslen=0):
    pWindow = np.ones(pWindowLen)
    sWindow = np.ones(sWindowLen)
    padLen = int((pWindowLen + 1) / 2)
    ppick = []
    spick = []
    pPredict= np.zeros((datap.shape[0], datap.shape[1]))   
    sPredict= np.zeros((datas.shape[0], datas.shape[1]))
    for i in range(datap.shape[0]):
        pPredict[i,:] = datap[i, :]###
        sPredict[i,:] = datas[i, :]###
    pRes = np.zeros((datap.shape[0], datap.shape[1]))
    sRes = np.zeros((datap.shape[0], datap.shape[1]))
    threads_per_block_x = 64
    threads_per_block_y = (datap.shape[0] + threads_per_block_x - 1) // threads_per_block_x
    blocks_per_grid_x = 64
    blocks_per_grid_y = (datap.shape[1] + blocks_per_grid_x - 1) // blocks_per_grid_x
    griddim = (blocks_per_grid_x, blocks_per_grid_y)
    blockdim = (threads_per_block_x,threads_per_block_y)
    pRescuda=cuda.to_device(pRes)
    rawnum=datap.shape[0]
    colnum=datap.shape[1]
    getres[griddim,blockdim](pRescuda, pPredict, pWindowLen, colnum ,rawnum)
    cuda.synchronize()
    pRes = pRescuda.copy_to_host()
    pRes=pPredict####
    ppick = []
    for j in range(datap.shape[0]):
        pPeakArray = sgn.find_peaks(pRes[j,:], height=height, distance=minDistance)
        if pPeakArray[0].size != 0:
            for h in range(pPeakArray[0].size):
                ppick.append(
                    [j % tracenum, j // tracenum, pPeakArray[0][h] / samplerate,
                     pPeakArray[1]['peak_heights'][h],
                     0])

    threads_per_block_x = 64
    threads_per_block_y = (datas.shape[0] + threads_per_block_x - 1) // threads_per_block_x
    blocks_per_grid_x = 64
    blocks_per_grid_y = (datas.shape[1] + blocks_per_grid_x - 1) // blocks_per_grid_x
    griddim = (blocks_per_grid_x, blocks_per_grid_y)
    blockdim = (threads_per_block_x,threads_per_block_y)
    sRescuda=cuda.to_device(sRes)
    rawnum=datas.shape[0]
    colnum=datas.shape[1]
    getres[griddim,blockdim](sRescuda, sPredict, sWindowLen, colnum ,rawnum)
    cuda.synchronize()
    sRes = sRescuda.copy_to_host()
    sRes=sPredict####
    spick = []
    for j in range(datas.shape[0]):
        sPeakArray = sgn.find_peaks(sRes[j,:], height=height, distance=minDistance)
        if sPeakArray[0].size != 0:
            for h in range(sPeakArray[0].size):
                spick.append(
                    [j % tracenum, j // tracenum, sPeakArray[0][h] / samplerate,
                     sPeakArray[1]['peak_heights'][h],
                     1])

    ppick = np.asarray(ppick)
    spick = np.asarray(spick)
    if ppick.shape[0] != 0 and spick.shape[0] != 0:
        pick = np.concatenate((ppick, spick), axis=0)
        return pick
    if ppick.shape[0] == 0 and spick.shape[0] != 0:
        return spick
    if spick.shape[0] == 0 and ppick.shape[0] != 0:
        return ppick
    else:
        print('null')
        return np.array([])


def gruvalid(pick, rcvpath=r'D:\TOC2ME\rcv_ddzbx.csv'):


    rcv = np.loadtxt(open(rcvpath, 'rb'), delimiter=',', skiprows=1, usecols=[0, 2, 3, 4])
    rcvNameList = [str(int(rcv[i, 0])) for i in range(rcv.shape[0])]
    rcvPositionList = rcv[:, 1:].astype(np.float32)
    Rec_Num = rcvPositionList.shape[0]
    xmax=rcv[:,1].max()-rcv[:,1].min()
    ymax = rcv[ :,2].max()-rcv[:,2].min()
    rcvid = []
    rcvpos = []
    for i in range(pick.shape[0]):
        rcvpos.append([pick[i, 0], int(rcvNameList[int(pick[i, 0])]), (rcvPositionList[int(pick[i, 0])][0]-rcv[:,1].min()) / xmax,
                       (rcvPositionList[int(pick[i, 0])][1]-rcv[:,2].min()) / ymax])  # id,name,x,y
    rcvpos = np.asarray(rcvpos)
    gruvalid = np.concatenate((rcvpos, pick[:, 2:3], pick[:, -1:], pick[:, -2:-1], pick[:, 1:2]),
                              axis=1)  # [id,name,x,y],t,phase,height,winindex
    return gruvalid


def getFilePathList(path, filetype):
    pathList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                pathList.append(file)
    return pathList

def valid_normalization(micdata,batchnum, inputsize,batchoverlap,reslen=0,tracenum=69):
    if  reslen!= 0:
        valid = np.zeros((int(micdata.shape[0] / 3 * (batchnum + 1)), inputsize, 3))
        for i in range(batchnum):
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 0] = micdata[0:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 1] = micdata[1:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 2] = micdata[2:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            for k in range(int(micdata.shape[0] / 3)):
                data_train = valid[i * int(tracenum) + k, :, :] * 1
                data_train = (data_train - np.mean(data_train))
                data_train = data_train / np.std(data_train)
                valid[i * int(tracenum) + k, :, :] = data_train
        valid[- int(tracenum):, :, 0] = micdata[0:tracenum*3:3, -inputsize:] * 1
        valid[- int(tracenum):, :, 1] = micdata[1:tracenum*3:3, -inputsize:] * 1
        valid[- int(tracenum):, :, 2] = micdata[2:tracenum*3:3, -inputsize:] * 1
        for k in range(int(micdata.shape[0] / 3)):
            data_train = valid[batchnum * int(tracenum) + k, :, :] * 1
            data_train = (data_train - np.mean(data_train))
            data_train = data_train / np.std(data_train)
            valid[batchnum * int(tracenum) + k, :, :] = data_train
    else:
        valid = np.zeros((int(micdata.shape[0] / 3) * (batchnum), micdata.shape[1], 3))
        for i in range(batchnum):
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 0] = micdata[0:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 1] = micdata[1:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            valid[i * int(tracenum):i * int(tracenum) + int(tracenum), :, 2] = micdata[2:tracenum*3:3,
                                                (inputsize - batchoverlap) * i:(
                                                                                        inputsize - batchoverlap) * i + inputsize] * 1
            for k in range(int(micdata.shape[0] / 3)):
                data_train = valid[i * int(tracenum) + k, :, :]
                data_train = (data_train - np.mean(data_train))
                data_train = data_train / np.std(data_train)
                valid[i * int(tracenum) + k, :, :] = data_train
    return valid
import argparse
def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--probability", default=0.25, help="phase probability")
    parser.add_argument("--minDistance", default=40)
    parser.add_argument("--rcvpath", default="./data/bcrcv.csv", help="Receive file directory")
    parser.add_argument("--divtime", default=3, help="Input csv file")
    parser.add_argument("--chosentime", default=1.2, help="Input hdf5 file")
    parser.add_argument("--wttPath", default='./data/wtt.npz', help="Traveltime file directory")
    parser.add_argument("--datapath",default='./data/bc.npy', help="Data file directory")
    parser.add_argument("--inputsize", default=15008, help="Inputsize of phase picking model")
    parser.add_argument("--batchoverlap", default=3008, type=int, help="Overlap length of phase picking ")
    parser.add_argument("--pamodeldir", default="./model/gru.hdf5", help="GRU model file directory")
    parser.add_argument("--winlen",default=1000, help="Length of phase association window")
    parser.add_argument("--savenpy", default=1, help="Whether to save the semblance file")
    parser.add_argument("--savenpyDir", default="./save", help="Semblance file output directory")
    parser.add_argument("--txtDir", default="./save.txt", help="Location results")
    parser.add_argument("--pickcut", default=12000, help="Input csv file")   
    parser.add_argument("--samplerate", default=100) 
    parser.add_argument("--threshold_semmax", default=0.2)  
    parser.add_argument("--threshold_numoflocalpeak", default=15) 
    parser.add_argument("--threshold_numofassociatedstation", default=2) 
    parser.add_argument("--threshold_rateoflocalpeak", default=0.8) 
    args = parser.parse_args()
    return args

def main(args):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7) 
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    stall = time.time()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    probability=args.probability
    minDistance=args.minDistance
    divtime = args.divtime
    chosentime = args.chosentime
    rcvpath=args.rcvpath
    wttPath=args.wttPath
    inputsize=args.inputsize
    batchoverlap = args.batchoverlap
    cut=args.pickcut
    samplerate = args.samplerate
    threshold_semmax=args.threshold_semmax
    threshold_numoflocalpeak=args.threshold_numoflocalpeak
    threshold_numofassociatedstation=args.threshold_numofassociatedstation
    localpeak_rate=args.threshold_rateoflocalpeak
    input_shape = (inputsize, 1,3)
    model = segunet(input_shape)
    model.load_weights('./model/segunet.hdf5', by_name=True)
    batch_size = 32
    modelpa = GRU_PA()
    modelpa.load_weights(args.pamodeldir)
    bc=np.load(args.datapath)[:,:]
    bc[39:42]=np.zeros((3,bc.shape[1]))
    global graph1
    global graph2
    graph1 = tf.get_default_graph()
    graph2 = tf.get_default_graph()
    savenpy=args.savenpy
    saveDirPath=args.savenpyDir
    txtpath=args.txtDir
    num=math.ceil(bc.shape[1]/cut)
    winlen = args.winlen
    starhour = 0
    startmin = 0
    startsec = 0
    skip = 0
    cuttime=cut/samplerate
    overlap = int(winlen / 2)
    padbatchnum=(inputsize-bc.shape[1]+cut*(num-1))//overlap
    padtime=(inputsize-bc.shape[1]+cut*(num-1))/samplerate
    if os.path.exists(txtpath):
        os.remove(txtpath)
    with open(txtpath, 'w') as f:
        f.write('X,Y,Z,semblance_max,semblance_min,localpeak_num,associated_station_num,min,sec,weight_p,weight_s,associated_phase_num,associated_p_num,associated_s_num\n')
    for ii in range(int(num)):
        micdata = bc[:,cut*ii:cut*ii+inputsize]
        if ii == num-1:
            micdata = bc[:,-inputsize:]
        miclen = micdata.shape[1]
        batch_overlap_frontnum = int((batchoverlap-overlap) // (winlen - overlap) / 2)  
        batch_overlap_behindnum = (batchoverlap-overlap) // (winlen - overlap) - batch_overlap_frontnum 
        batchwinnum = (inputsize - overlap) // (winlen - overlap)
        reslen  = 0
        tracenum=micdata.shape[0]/3
        pWindowLen=31
        sWindowLen=31
        batchnum = (micdata.shape[1] -batchoverlap)//(inputsize-batchoverlap)     
        if batchnum==0:
            batchnum=1      
        nn=batchnum
        wintime = winlen / samplerate
        batchsemsize = (batchwinnum - batch_overlap_frontnum - batch_overlap_behindnum) * (winlen - overlap) + overlap
        batchsemtime = batchsemsize / samplerate
        batchtime = inputsize / samplerate
        valid=valid_normalization(micdata,batchnum, inputsize,batchoverlap,tracenum=int(tracenum))
        if valid.shape[1]!=inputsize:
            print('------------valid.shape[1]!=inputsize-----------',ii)
            validpadlen=inputsize-micdata.shape[1]
            valid=np.pad(valid,((0,0),(0,validpadlen),(0,0)),'constant', constant_values=(0,0))
        valid=valid[:,:,np.newaxis,:]
        stpre = time.time()                
        with graph1.as_default():
            y_pre = model.predict(valid, batch_size=batch_size, verbose=0)
        p = np.squeeze(y_pre[:, :, 0])
        s = np.squeeze(y_pre[:, :, 1])
        print(time.time() - stpre)
        stpick = time.time()
        pick = pickpoint(p, s, samplerate=samplerate,
                minDistance=minDistance, tracenum=tracenum,
                pWindowLen=pWindowLen, sWindowLen=sWindowLen,
                wintime=wintime, height=probability)
        if pick.shape[0]==0:
            continue
        print(time.time() - stpick)
        grudata = gruvalid(pick, rcvpath=rcvpath)
        grusort = np.zeros((1, 8))
        stpha = time.time()
        for i in range(nn):
            if np.argwhere(grudata[:, -1] == i).shape[0] != 0:
                grudata1 = grudata[np.argwhere(grudata[:, -1] == i)[:, 0], :] * 1
                if grudata1.shape[0] != 1:
                    grudata1 = grudata1[np.argsort([grudata1[:, 4]])[0, :], :]# id, name,x,y,t,phase,height,winindex
                    grusort = np.concatenate((grusort, grudata1), axis=0)
                else:
                    grudata1 = grudata1[np.argsort([grudata1[:, 4]])[0, :], :]# id, name,x,y,t,phase,height,winindex
                    grusort = np.concatenate((grusort, grudata1), axis=0)
        grusort = np.delete(grusort, 0, axis=0)
        gruout = np.zeros((1, 8))
        for j in range(batchnum):
            if ii == 0:
                if np.argwhere(grusort[:, -1] == j).shape[0] != 0:
                    gru = grusort[np.argwhere(grusort[:, -1] == j)[:, 0], :]
                    for i in range(batchwinnum - batch_overlap_behindnum):
                        if np.argwhere(np.logical_and(gru[:, 4] > i * wintime - i * overlap / samplerate,
                                                    gru[:, 4] < (i + 1) * wintime - i * overlap / samplerate)).shape[
                            0] < threshold_numofassociatedstation*2:
                            continue
                        window = np.argwhere(np.logical_and(gru[:, 4] > i * wintime - i * overlap / samplerate,
                                                            gru[:, 4] < (i + 1) * wintime - i * overlap / samplerate))
                        windowlast = window[-1, 0] + 1
                        windowfirst = window[0, 0]
                        gruwin = gru[np.newaxis, windowfirst:windowlast,:] * 1  # id, name,x,y,t,phase,height,winindex
                        gruwin[:, :, 4] = gruwin[:, :, 4] - i * (wintime - overlap / samplerate)
                        with graph2.as_default():
                            pa = modelpa.predict(gruwin[:, :, 2:-2], verbose=1)
                        gruwin[:, :, 4] = gruwin[:, :, 4] + i * (wintime - overlap / samplerate)
                        index = np.ones((pa.shape[1], 1)) * (i)
                        gruout1 = np.concatenate((gruwin[0, :, :1], gruwin[0, :, 4:], index[:, :],
                                                pa[0, :]),
                                                axis=1)  # id ,t phase hei winindex,index,true false
                        gruout = np.concatenate((gruout, gruout1), axis=0)
            if ii == num-1:
                if np.argwhere(grusort[:, -1] == j).shape[0] != 0:
                    gru = grusort[np.argwhere(grusort[:, -1] == j)[:, 0], :]
                    for i in range(batchwinnum - batch_overlap_frontnum-padbatchnum):
                        if np.argwhere(np.logical_and(gru[:, 4] > (i+ batch_overlap_frontnum + padbatchnum)* 
                                wintime - (i+ batch_overlap_frontnum+padbatchnum) * overlap / samplerate,
                                gru[:, 4] < (i + batch_overlap_frontnum+padbatchnum+ 1) * wintime - 
                                (i+ batch_overlap_frontnum+padbatchnum) * overlap / samplerate)).shape[0] < threshold_numofassociatedstation*2:
                            continue
                        window = np.argwhere(np.logical_and(gru[:, 4] > (i+ batch_overlap_frontnum+ padbatchnum ) * wintime - 
                                                (i+ batch_overlap_frontnum+ padbatchnum) * overlap / samplerate,
                                            gru[:, 4] < ((i+ batch_overlap_frontnum+ padbatchnum) + 1) * wintime - 
                                                (i+ batch_overlap_frontnum+ padbatchnum) * overlap / samplerate))
                        windowlast = window[-1, 0] + 1
                        windowfirst = window[0, 0]
                        gruwin = gru[np.newaxis, windowfirst:windowlast,:] * 1  # id, name,x,y,t,phase,height,winindex
                        gruwin[:, :, 4] = gruwin[:, :, 4] - (i+ batch_overlap_frontnum+ padbatchnum) * (wintime - overlap / samplerate)
                        with graph2.as_default():
                            pa = modelpa.predict(gruwin[:, :, 2:-2], verbose=1)
                        gruwin[:, :, 4] = gruwin[:, :, 4] + (i+ batch_overlap_frontnum+ padbatchnum) * (wintime - overlap / samplerate)
                        index = np.ones((pa.shape[1], 1)) * (i)
                        gruout1 = np.concatenate((gruwin[0, :, :1], gruwin[0, :, 4:], index[:, :],pa[0, :]),axis=1)  # id ,t phase hei winindex,index,true false
                        gruout = np.concatenate((gruout, gruout1), axis=0)
            if ii!=num-1 and ii!=0:
                if np.argwhere(grusort[:, -1] == j).shape[0] != 0:
                    gru = grusort[np.argwhere(grusort[:, -1] == j)[:, 0], :]
                    for i in range(batchwinnum - batch_overlap_frontnum - batch_overlap_behindnum):
                        if np.argwhere(np.logical_and(gru[:, 4] > (i+batch_overlap_frontnum) * wintime - 
                                        (i+batch_overlap_frontnum) * overlap / samplerate,
                                        gru[:, 4] < (i+batch_overlap_frontnum + 1) * wintime 
                                        - (i+batch_overlap_frontnum) * overlap / samplerate)).shape[0] < threshold_numofassociatedstation*2:
                            continue
                        window = np.argwhere(np.logical_and(gru[:, 4] > (i+batch_overlap_frontnum) * wintime - (i+batch_overlap_frontnum) * overlap / samplerate,
                                                gru[:, 4] < (i+batch_overlap_frontnum + 1) * wintime -(i+batch_overlap_frontnum)  * overlap / samplerate))
                        windowlast = window[-1, 0] + 1
                        windowfirst = window[0, 0]
                        gruwin = gru[np.newaxis, windowfirst:windowlast,
                                :] * 1  # id, name,x,y,t,phase,height,index
                        gruwin[:, :, 4] = gruwin[:, :, 4] - (i+batch_overlap_frontnum) * (wintime - overlap / samplerate)
                        with graph2.as_default():
                            pa = modelpa.predict(gruwin[:, :, 2:-2], verbose=1)
                        gruwin[:, :, 4] = gruwin[:, :, 4] + (i+batch_overlap_frontnum) * (wintime - overlap / samplerate)
                        index = np.ones((pa.shape[1], 1)) * (i)
                        gruout1 = np.concatenate((gruwin[0, :, :1], gruwin[0, :, 4:], index[:,:],
                                                pa[0, :]),
                                                axis=1)  # id ,t phase hei winindex,index,true false
                        gruout = np.concatenate((gruout, gruout1), axis=0)
        gruout = np.delete(gruout, 0, axis=0)
        print(time.time() - stpha)
        with open(txtpath, 'ab') as f:
            for h in range(batchnum):
                if ii==0:
                    pa=gruout[np.argwhere(gruout[:, 4] == h)[:, 0], :] * 1
                    for i in range(batchwinnum-batch_overlap_behindnum):
                        index =  i
                        pabatch = pa[np.argwhere(pa[:, 5] == index)[:, 0], :] * 1
                        pabatchbhd = pa[np.argwhere(pa[:, 5] == index + 1)[:, 0], :] * 1
                        if pabatch.shape[0] == 0 or skip == 1:
                            skip = 0
                            continue
                        if np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7])).shape[0] > 0 \
                                and np.argwhere(np.logical_and(pabatch[:, 2] == 1, pabatch[:, 6] > pabatch[:, 7])).shape[0] > 0 \
                                and np.argwhere(pabatch[:, 6] > pabatch[:, 7]).shape[0] > (pabatch.shape[0]) / 3:
                            paptrue = pabatch[np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7]))[:, 0], 1]
                            if pabatchbhd.shape[0] != 0:
                                if np.argwhere(
                                        np.logical_and(pabatchbhd[:, 2] == 0,pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[0] > 0 \
                                        and np.argwhere(np.logical_and(pabatchbhd[:, 2] == 1,pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[0] > 0 \
                                        and np.argwhere(pabatchbhd[:, 6] > pabatchbhd[:, 7]).shape[0] > (pabatchbhd.shape[0]) / 3:
                                    papbhdtrue = pabatchbhd[np.argwhere(np.logical_and(
                                                                        pabatchbhd[:, 2] == 0,
                                                                        pabatchbhd[:, 6] > pabatchbhd[:,7]))[:, 0], 1]
                                    tbhd = np.mean(papbhdtrue)
                                    t = np.mean(paptrue)
                                    if np.abs(t - tbhd) < divtime:
                                        if t- i * (wintime - overlap / samplerate) > wintime - overlap / samplerate + chosentime:
                                            continue
                                        skip = 1
                            pabatch[:, 1] = pabatch[:, 1] - i * (wintime - overlap / samplerate)
                            pap = pabatch[np.argwhere(pabatch[:, 2] == 0)[:, 0], :] * 1
                            pap = pap[np.argwhere(pap[:, -2] > pap[:, -1])[:, 0], :]
                            micp=[]
                            for pp in range(pap.shape[0]):
                                if int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap)<0:
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,0:
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                else:    
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap):
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                micp.append(abs(micdatap).max())
                            micparray=np.array(micp)
                            papfir = int(pap[np.argmin(pap[:, 1]), 1] * samplerate) - 100
                            paplas = int(pap[np.argmax(pap[:, 1]), 1] * samplerate) + 100
                            refIndexP = int(pap[(pap[:, 3].argmax(axis=0)), 0])
                            refPointP = int(pap[(pap[:, 3].argmax(axis=0)), 1] * samplerate) - papfir

                            pas = pabatch[np.argwhere(pabatch[:, 2] == 1)[:, 0], :]
                            pas = pas[np.argwhere(pas[:, -2] > pas[:, -1])[:,0], :]
                            mics=[]
                            for ss in range(pas.shape[0]):
                                if int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap)<0:
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())  
                                else:                            
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())
                            micsarray=np.array(mics)
                            pasfir = int(pas[np.argmin(pas[:, 1]), 1] * samplerate) - 100
                            paslas = int(pas[np.argmax(pas[:, 1]), 1] * samplerate) + 100
                            refIndexS = int(pas[(pas[:, 3].argmax(axis=0)), 0])
                            refPointS = int(pas[(pas[:, 3].argmax(axis=0)), 1] * samplerate) - pasfir
                            weip=np.nanmedian( micparray)
                            weis=np.nanmedian( micsarray)
                            weightp=weip/(weip+weis)
                            weights=weis/(weip+weis)
                            txt, semmax,LocPeakNum,AssociatedStationNum = sembalance_ap_PSGPU(
                                p[h * int(tracenum):h * int(tracenum) + int(tracenum), i * (winlen - overlap) + papfir:i * (winlen - overlap) + paplas],
                                s[h * int(tracenum):h * int(tracenum) + int(tracenum), i * (winlen - overlap) + pasfir:i * (winlen - overlap) + paslas],
                                pabatch,  wttPath=wttPath,
                                saveDirPath=saveDirPath, samplingRate=samplerate,
                                windowLen=50, refPointP=refPointP, refPointS=refPointS, refIndexP=refIndexP,
                                refIndexS=refIndexS, wintime=wintime, winno=i, overlap=overlap / samplerate,
                                semName= '%d' % (ii*cuttime+(wintime*i/2)//60)+ '_'+'%d' % ((wintime*i/2)%60) ,
                                savenpy=savenpy,weightp=weightp,weights=weights,rate=localpeak_rate )
                            if semmax > threshold_semmax and LocPeakNum<=threshold_numoflocalpeak \
                                and AssociatedStationNum>=threshold_numofassociatedstation:                                        
                                paindex = np.asarray(([pabatch.shape[0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 0,
                                                                                pabatch[:, 6] > pabatch[:, 7])).shape[
                                                        0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 1,
                                                                                pabatch[:, 6] > pabatch[:, 7])).shape[
                                                        0]]
                                                    ))
                                sourcesec = (wintime*i/2)%60
                                sourcemin = (ii*cuttime+(wintime*i/2))//60
                                txt = np.concatenate((txt, np.asarray([sourcemin, sourcesec, weightp,weights]), paindex))
                                txt = txt[np.newaxis, :]
                                np.savetxt(f, txt, delimiter=',')
                if ii==num-1:
                    pa = gruout[np.argwhere(gruout[:, 4] == h)[:, 0], :] * 1
                    for i in range(batchwinnum - batch_overlap_frontnum-padbatchnum):
                        index = i
                        pabatch = pa[np.argwhere(pa[:, 5] == index)[:, 0], :] * 1
                        pabatchbhd = pa[np.argwhere(pa[:, 5] == index + 1)[:, 0], :] * 1      
                        if pabatch.shape[0] == 0 or skip == 1:
                            skip = 0                
                            continue               
                        if np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7])).shape[
                            0] > 0 and \
                                np.argwhere(
                                    np.logical_and(pabatch[:, 2] == 1, pabatch[:, 6] > pabatch[:, 7])).shape[
                                    0] > 0 and \
                                np.argwhere(pabatch[:, 6] > pabatch[:, 7]).shape[0] > (pabatch.shape[0]) / 3:
                            print(index)                              
                            paptrue = pabatch[np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7]))[:,0], 1]
                            if pabatchbhd.shape[0] != 0:
                                if np.argwhere(
                                        np.logical_and(pabatchbhd[:, 2] == 0,
                                                    pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[
                                    0] > 1 and \
                                        np.argwhere(np.logical_and(pabatchbhd[:, 2] == 1,
                                                                pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[
                                            0] > 1 and \
                                        np.argwhere(pabatchbhd[:, 6] > pabatchbhd[:, 7]).shape[0] > (pabatchbhd.shape[0]) / 3:
                                    papbhdtrue = pabatchbhd[np.argwhere(np.logical_and(pabatchbhd[:, 2] == 0,
                                                                pabatchbhd[:,6] > pabatchbhd[:,7]))[:, 0], 1]
                                    tbhd = np.mean(papbhdtrue)
                                    t = np.mean(paptrue)
                                    if np.abs(t - tbhd) < divtime:
                                        if t- (i+batch_overlap_frontnum+padbatchnum) * (wintime - overlap / samplerate) > wintime - overlap / samplerate + chosentime:
                                            continue
                                        skip = 1
                            pabatch[:, 1] = pabatch[:, 1] - (i+batch_overlap_frontnum+padbatchnum) * (wintime - overlap / samplerate)# [index,time,phase,height,true,false]
                            pap = pabatch[np.argwhere(pabatch[:, 2] == 0)[:, 0], :] * 1
                            pap = pap[np.argwhere(pap[:, -2] > pap[:, -1])[:, 0], :]
                            micp=[]
                            for pp in range(pap.shape[0]):
                                if int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap)<0:
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,0:
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                else:    
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap):
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                micp.append(abs(micdatap).max())
                            micparray=np.array(micp)
                            papfir = int(pap[np.argmin(pap[:, 1]), 1] * samplerate) - 100
                            paplas = int(pap[np.argmax(pap[:, 1]), 1] * samplerate) + 100
                            refIndexP = int(pap[(pap[:, 3].argmax(axis=0)), 0])
                            refPointP = int(pap[(pap[:, 3].argmax(axis=0)), 1] * samplerate) - papfir

                            pas = pabatch[np.argwhere(pabatch[:, 2] == 1)[:, 0], :]
                            pas = pas[np.argwhere(pas[:, -2] > pas[:, -1])[:,0], :]
                            mics=[]
                            for ss in range(pas.shape[0]):
                                if int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap)<0:
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())  
                                else:                            
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())
                            micsarray=np.array(mics)
                            pasfir = int(pas[np.argmin(pas[:, 1]), 1] * samplerate) - 100
                            paslas = int(pas[np.argmax(pas[:, 1]), 1] * samplerate) + 100
                            refIndexS = int(pas[(pas[:, 3].argmax(axis=0)), 0])
                            refPointS = int(pas[(pas[:, 3].argmax(axis=0)), 1] * samplerate) - pasfir
                            weip=np.nanmedian( micparray)
                            weis=np.nanmedian( micsarray)
                            weightp=weip/(weip+weis)
                            weights=weis/(weip+weis)   
                            txt, semmax,LocPeakNum,AssociatedStationNum = sembalance_ap_PSGPU(
                                p[h * int(tracenum):h * int(tracenum) + int(tracenum), (i+batch_overlap_frontnum+padbatchnum) * (winlen - overlap) 
                                    + papfir:(i+batch_overlap_frontnum+padbatchnum) * (winlen - overlap) + paplas],
                                s[h * int(tracenum):h * int(tracenum) + int(tracenum), (i+batch_overlap_frontnum+padbatchnum) * (winlen - overlap) 
                                    + pasfir:(i+batch_overlap_frontnum+padbatchnum) * (winlen - overlap) + paslas],
                                pabatch,  wttPath=wttPath,
                                saveDirPath=saveDirPath, samplingRate=samplerate,
                                windowLen=50, refPointP=refPointP, refPointS=refPointS, refIndexP=refIndexP,
                                refIndexS=refIndexS, wintime=wintime, winno=i, overlap=overlap / samplerate,
                                semName= '%d' % ((ii*cuttime-padtime+(wintime*(i+batch_overlap_frontnum+padbatchnum)/2))//60)
                                    + '_'+'%d' % (-padtime+(wintime*(i+batch_overlap_frontnum+padbatchnum)/2)%60) ,
                                savenpy=savenpy,weightp=weightp,weights=weights,rate=localpeak_rate)
                            if semmax > threshold_semmax and LocPeakNum<=threshold_numoflocalpeak \
                                and AssociatedStationNum>=threshold_numofassociatedstation: 
                                paindex = np.asarray(([pabatch.shape[0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 0,
                                                                                pabatch[:, 6] > pabatch[:,7])).shape[0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 1,
                                                                                pabatch[:, 6] > pabatch[:,7])).shape[0]]
                                                    ))
                                sourcesec = (-padtime+wintime*(i+batch_overlap_frontnum+padbatchnum)/2)%60
                                sourcemin = (ii*cuttime-padtime+(wintime*(i+batch_overlap_frontnum+padbatchnum)/2))//60
                                txt = np.concatenate((txt, np.asarray([sourcemin, sourcesec,weightp,weights]), paindex))
                                txt = txt[np.newaxis, :]

                                np.savetxt(f, txt, delimiter=',')
                else:
                    pa = gruout[np.argwhere(gruout[:, 4] == h)[:, 0], :] * 1
                    for i in range(batchwinnum - batch_overlap_frontnum-batch_overlap_behindnum):
                        index = i
                        pabatch = pa[np.argwhere(pa[:, 5] == index)[:, 0], :] * 1
                        pabatchbhd = pa[np.argwhere(pa[:, 5] == index + 1)[:, 0], :] * 1      

                        if pabatch.shape[0] == 0 or skip == 1:
                            skip = 0                
                            continue               
                        if np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7])).shape[
                            0] > 0 and \
                                np.argwhere(
                                    np.logical_and(pabatch[:, 2] == 1, pabatch[:, 6] > pabatch[:, 7])).shape[
                                    0] > 0 and \
                                np.argwhere(pabatch[:, 6] > pabatch[:, 7]).shape[0] > (pabatch.shape[0]) / 3:                           
                            paptrue = pabatch[np.argwhere(np.logical_and(pabatch[:, 2] == 0, pabatch[:, 6] > pabatch[:, 7]))[:,0], 1]
                            if pabatchbhd.shape[0] != 0:
                                if np.argwhere(
                                        np.logical_and(pabatchbhd[:, 2] == 0,
                                                    pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[
                                    0] > 1 and \
                                        np.argwhere(np.logical_and(pabatchbhd[:, 2] == 1,
                                                                pabatchbhd[:, 6] > pabatchbhd[:, 7])).shape[
                                            0] > 1 and \
                                        np.argwhere(pabatchbhd[:, 6] > pabatchbhd[:, 7]).shape[0] > (pabatchbhd.shape[0]) / 3:
                                    papbhdtrue = pabatchbhd[np.argwhere(np.logical_and(pabatchbhd[:, 2] == 0,
                                                                pabatchbhd[:,6] > pabatchbhd[:,7]))[:, 0], 1]
                                    tbhd = np.mean(papbhdtrue)
                                    t = np.mean(paptrue)
                                    if np.abs(t - tbhd) < divtime:
                                        if t- (i+batch_overlap_frontnum) * (wintime - overlap / samplerate) > wintime - overlap / samplerate + chosentime:
                                            continue
                                        skip = 1
                            pabatch[:, 1] = pabatch[:, 1] - (i+batch_overlap_frontnum) * (wintime - overlap / samplerate)# [index,time,phase,height,true,false]
                            pap = pabatch[np.argwhere(pabatch[:, 2] == 0)[:, 0], :] * 1
                            pap = pap[np.argwhere(pap[:, -2] > pap[:, -1])[:, 0], :]
                            micp=[]
                            for pp in range(pap.shape[0]):
                                if int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap)<0:
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,0:
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                else:    
                                    micdatap=micdata[int(pap[pp,0])*3+2:int(pap[pp,0])*3+3,int(pap[pp,1]* samplerate)-int(0.15* samplerate)+i * (winlen - overlap):
                                    int(pap[pp,1]* samplerate)+int(0.15* samplerate)+i * (winlen - overlap)]
                                micp.append(abs(micdatap).max())
                            micparray=np.array(micp)
                            papfir = int(pap[np.argmin(pap[:, 1]), 1] * samplerate) - 100
                            paplas = int(pap[np.argmax(pap[:, 1]), 1] * samplerate) + 100
                            refIndexP = int(pap[(pap[:, 3].argmax(axis=0)), 0])
                            refPointP = int(pap[(pap[:, 3].argmax(axis=0)), 1] * samplerate) - papfir

                            pas = pabatch[np.argwhere(pabatch[:, 2] == 1)[:, 0], :]
                            pas = pas[np.argwhere(pas[:, -2] > pas[:, -1])[:,0], :]
                            mics=[]
                            for ss in range(pas.shape[0]):
                                if int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap)<0:
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,0:
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())  
                                else:                            
                                    micdatas1=micdata[int(pas[ss,0])*3:int(pas[ss,0])*3+1,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas1).max())
                                    micdatas2=micdata[int(pas[ss,0])*3+1:int(pas[ss,0])*3+2,int(pas[ss,1]* samplerate)-int(0.3* samplerate)+i * (winlen - overlap):
                                    int(pas[ss,1]* samplerate)+int(0.3* samplerate)+i * (winlen - overlap)]
                                    mics.append(abs(micdatas2).max())
                            micsarray=np.array(mics)
                            pasfir = int(pas[np.argmin(pas[:, 1]), 1] * samplerate) - 100
                            paslas = int(pas[np.argmax(pas[:, 1]), 1] * samplerate) + 100
                            refIndexS = int(pas[(pas[:, 3].argmax(axis=0)), 0])
                            refPointS = int(pas[(pas[:, 3].argmax(axis=0)), 1] * samplerate) - pasfir
                            weip=np.nanmedian( micparray)
                            weis=np.nanmedian( micsarray)
                            weightp=weip/(weip+weis)
                            weights=weis/(weip+weis)   
                            txt, semmax,LocPeakNum,AssociatedStationNum = sembalance_ap_PSGPU(
                                p[h * int(tracenum):h * int(tracenum) + int(tracenum), (i+batch_overlap_frontnum) * (winlen - overlap) + papfir:(i+batch_overlap_frontnum) * (winlen - overlap) + paplas],
                                s[h * int(tracenum):h * int(tracenum) + int(tracenum), (i+batch_overlap_frontnum) * (winlen - overlap) + pasfir:(i+batch_overlap_frontnum) * (winlen - overlap) + paslas],
                                pabatch,  wttPath=wttPath,
                                saveDirPath=saveDirPath, samplingRate=samplerate,
                                windowLen=50, refPointP=refPointP, refPointS=refPointS, refIndexP=refIndexP,
                                refIndexS=refIndexS, wintime=wintime, winno=i, overlap=overlap / samplerate,
                                semName= '%d' % ((ii*cuttime+(wintime*(i+batch_overlap_frontnum)/2))//60)+ '_'+'%d' % ((wintime*(i+batch_overlap_frontnum)/2)%60) ,
                                savenpy=savenpy,weightp=weightp,weights=weights,rate=localpeak_rate)

                            if semmax > threshold_semmax and LocPeakNum<=threshold_numoflocalpeak \
                                and AssociatedStationNum>=threshold_numofassociatedstation: 
                                paindex = np.asarray(([pabatch.shape[0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 0,
                                                                                pabatch[:, 6] > pabatch[:,7])).shape[0],
                                                    np.argwhere(np.logical_and(pabatch[:, 2] == 1,
                                                                                pabatch[:, 6] > pabatch[:,7])).shape[0]]
                                                    ))
                                sourcesec = (wintime*(i+batch_overlap_frontnum)/2)%60
                                sourcemin = (ii*cuttime+(wintime*(i+batch_overlap_frontnum)/2))//60
                                txt = np.concatenate((txt, np.asarray([sourcemin, sourcesec,weightp,weights]), paindex))
                                txt = txt[np.newaxis, :]

                                np.savetxt(f, txt, delimiter=',')
        del p
        del s
    print(time.time() - stall, '总用时')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    args = read_args()
    main(args)
