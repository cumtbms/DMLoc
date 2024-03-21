import numpy as np

import matplotlib.pyplot as plt

import copy
import os
import time
from numba import cuda
import scipy.signal as sgn

@cuda.jit
def apsemCUDAPS(X_Num, Y_Num, Z_Num, custartposition, deltx, delty,
                deltz, Src_Num, deltatime, weight,
                samplingRate, windowLen,
                refPoint, refTrace, apDataArray, samplingNum,
                semArray, positionArray, sUp,
                currentMicSingal, source_position, num):
    sourcethrxno = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if sourcethrxno < Src_Num:
        zno = sourcethrxno // (X_Num * Y_Num)
        yno = (sourcethrxno - (X_Num * Y_Num) * zno) // X_Num
        xno = sourcethrxno - (X_Num * Y_Num) * zno - X_Num * yno
        sxyz_x = custartposition[0] + xno * deltx
        sxyz_y = custartposition[1] + yno * delty
        sxyz_z = custartposition[2] + zno * deltz
        source_position[sourcethrxno, 0] = sxyz_x
        source_position[sourcethrxno, 1] = sxyz_y
        source_position[sourcethrxno, 2] = sxyz_z
        sDown = 0
        Weight = 0
        for i in range(num):
            micIndex = int(i)
            wttIndex = int(i)
            invertLen = int(deltatime[sourcethrxno, wttIndex] * samplingRate)
            if refPoint + invertLen + windowLen/2 > samplingNum or refPoint + invertLen -windowLen/2< 0:
                continue
            for j in range(windowLen):
                currentMicSingal[sourcethrxno, j] = apDataArray[int(micIndex)][refPoint + invertLen-int(windowLen/2) + j]
                sUp[sourcethrxno, j] = sUp[sourcethrxno, j] + currentMicSingal[sourcethrxno, j]
            for h in range(windowLen):
                sDown = sDown + currentMicSingal[sourcethrxno, h] * currentMicSingal[sourcethrxno, h]
                Weight = Weight + currentMicSingal[sourcethrxno, h]
        a = 0
        for k in range(windowLen):
            a = a + sUp[sourcethrxno, k] * sUp[sourcethrxno, k]
        sDown = (i + 1) * sDown
        xSourceIndex, ySourceIndex, zSourceIndex = int(xno), int(yno), int(zno)
        semArray[xSourceIndex, ySourceIndex, zSourceIndex] = a / sDown
        positionArray[xSourceIndex, ySourceIndex, zSourceIndex, 0] = source_position[sourcethrxno, 0]
        positionArray[xSourceIndex, ySourceIndex, zSourceIndex, 1] = source_position[sourcethrxno, 1]
        positionArray[xSourceIndex, ySourceIndex, zSourceIndex, 2] = source_position[sourcethrxno, 2]
        weight[xSourceIndex, ySourceIndex, zSourceIndex] = Weight


def sembalance_ap_PSGPU(
        apDataArrayP,
        apDataArrayS,
        appa,
        wttPath,
        saveDirPath='/data3/lyz/自动定位/microlab/npy/1031',
        samplingRate=500,
        wintime=6.144,
        winno=0,
        overlap=2,
        windowLen=50,
        refPointP=0,
        refPointS=0,
        refIndexP=0,
        refIndexS=0,
        suffix='-sem',
        semName='1',
        savenpy=0,
        weightp=1,
        weights=1,
        rate=0.8

):
    timestart = time.time()
    traveltime = np.load(wttPath)
    rcvNameList = traveltime['rcvNameList']  # 检波器name
    dxP = dyP = dzP = int(traveltime['delta'].astype('uint32'))

    startPosition = traveltime['startPosition']
    xMinP, yMinP, zMinP = startPosition[0], startPosition[1], startPosition[2]
    sourcenum = traveltime['sourcenum']
    xNumP, yNumP, zNumP = sourcenum[0], sourcenum[1], sourcenum[2]

    timearrayP = traveltime['ptime']
    timearrayS = traveltime['stime']

    apDataArrayP[np.isnan(apDataArrayP)] = 0
    apDataArrayS[np.isnan(apDataArrayS)] = 0
    apDataArrayP = np.ascontiguousarray(apDataArrayP)
    apDataArrayS = np.ascontiguousarray(apDataArrayS)
    # source_position.flags['C_CONTIGUOUS']

    tracesLen = apDataArrayP.shape[0]
    samplingNumP = apDataArrayP.shape[1]
    samplingNumS = apDataArrayS.shape[1]


    xNum = xNumP
    yNum = yNumP
    zNum = zNumP
    dx = dxP
    dy = dyP
    dz = dzP
    # 计算叠加系数
    semArrayP = np.zeros((xNum, yNum, zNum))
    Pweight = np.zeros((xNum, yNum, zNum))
    positionArrayP = np.zeros((xNum, yNum, zNum, 3))
    threads_per_block = 64
    blocks_per_grid_x = (xNum * yNum * zNum + threads_per_block - 1) // threads_per_block
    blocks_per_grid_y = 1
    griddim = (blocks_per_grid_x, blocks_per_grid_y)
    blockdim = (threads_per_block, 1)
    startpositionp = np.array([xMinP, yMinP, zMinP]).astype(np.float32)
    startpositions = np.array([xMinP, yMinP, zMinP]).astype(np.float32)
    Src_Num = xNum * yNum * zNum  # int(tracesLen/3)

    custartposition = cuda.to_device(startpositionp)
    cusemArrayP = cuda.to_device(semArrayP)
    cupositionArrayP = cuda.to_device(positionArrayP)
    cuPweight = cuda.to_device(Pweight)
    sup = np.zeros((xNum * yNum * zNum, windowLen))
    currentMicSingal = np.zeros((xNum * yNum * zNum, windowLen))
    sup = cuda.to_device(sup)
    currentMicSingal = cuda.to_device(currentMicSingal)
    source_position = np.zeros((xNum * yNum * zNum, 3))
    source_position = cuda.to_device(source_position)
    deltatimeP = timearrayP - timearrayP[:, refIndexP:refIndexP + 1]
    apsemCUDAPS[griddim, blockdim](xNum, yNum, zNum, custartposition, dx,
                                   dy, dz, Src_Num, deltatimeP, cuPweight,
                                   samplingRate, windowLen, refPointP,
                                   refIndexP, apDataArrayP, samplingNumP, cusemArrayP,
                                   cupositionArrayP, sup, currentMicSingal, source_position, tracesLen)
    print(time.time() - timestart, 'p')
    cuda.synchronize()
    semArrayP = cusemArrayP.copy_to_host()
    positionArray = cupositionArrayP.copy_to_host()
    Pweight = cuPweight.copy_to_host()
    xLocationIndexP, yLocationIndexP, zLocationIndexP = np.where(semArrayP == np.max(semArrayP))

    semArrayS = np.zeros((xNum, yNum, zNum))
    positionArrayS = np.zeros((xNum, yNum, zNum, 3))
    Sweight = np.zeros((xNum, yNum, zNum))

    cusemArrayS = cuda.to_device(semArrayS)
    custartpositions = cuda.to_device(startpositions)
    cupositionArrayS = cuda.to_device(positionArrayS)
    cuSweight = cuda.to_device(Sweight)
    sup = np.zeros((xNum * yNum * zNum, windowLen))
    currentMicSingal = np.zeros((xNum * yNum * zNum, windowLen))
    sup = cuda.to_device(sup)
    currentMicSingal = cuda.to_device(currentMicSingal)
    source_position = np.zeros((xNum * yNum * zNum, 3))
    source_position = cuda.to_device(source_position)
    deltatimeS = timearrayS - timearrayS[:, refIndexS:refIndexS + 1]
    apsemCUDAPS[griddim, blockdim](xNum, yNum, zNum, custartpositions, dx,
                                   dy, dz, Src_Num, deltatimeS, cuSweight,
                                   samplingRate, windowLen, refPointS,
                                   refIndexS, apDataArrayS, samplingNumS, cusemArrayS,
                                   cupositionArrayS, sup, currentMicSingal, source_position, tracesLen)
    print(time.time() - timestart, 's')
    cuda.synchronize()
    semArrayS = cusemArrayS.copy_to_host()
    positionArray = cupositionArrayS.copy_to_host()
    Sweight = cuSweight.copy_to_host()
    xLocationIndexS, yLocationIndexS, zLocationIndexS = np.where(semArrayS == np.max(semArrayS))

    Pweight = sum(sum(sum(Pweight)))
    Sweight = sum(sum(sum(Sweight)))
    sumweight = Pweight + Sweight
    semArray = semArrayP * weightp + semArrayS * weights
    xLocationIndex, yLocationIndex, zLocationIndex = np.where(semArray == np.max(semArray))

#peak
    ax=np.array([[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[0,1,1],[0,1,-1],[0,-1,-1],[0,-1,1], \
                [1,0,1],[1,0,-1],[1,1,0],[1,-1,0],[1,1,1],[1,1,-1],[1,-1,-1],[1,-1,1],[1,0,0],\
                [-1,0,1],[-1,0,-1],[-1,1,0],[-1,-1,0],[-1,1,1],[-1,1,-1],[-1,-1,-1],[-1,-1,1],[-1,0,0]])
    panpos = np.array([np.array([xLocationIndex, yLocationIndex, zLocationIndex])[0, 0],np.array([xLocationIndex, yLocationIndex, zLocationIndex])[1, 0],
        np.array([xLocationIndex, yLocationIndex, zLocationIndex])[2, 0]])
    pan = semArray
    appatrue=appa[np.argwhere(appa[:,6]>appa[:,7])[:,0],:]
    appatruep=appatrue[np.argwhere(appatrue[:,2]==0)[:,0],:]
    appatrues=appatrue[np.argwhere(appatrue[:,2]==1)[:,0],:]
    num=0
    for i in range(tracesLen):
        if ( appatruep[:,0]==i).any() and ( appatrues[:,0]==i).any():
            num=num+1
    panmax=(pan.max()).max()
    kk=[]
    for i in range(1,pan.shape[0]-1):
        pany=np.concatenate((pan[i,:,:],np.zeros((pan.shape[1],1))),axis=1)
        yfla=pany.flatten('a')
        peaky=sgn.find_peaks(yfla, height=0, distance=1)
        yindex = np.asarray(peaky[0])
        panz=np.concatenate((pan[i,:,:],np.zeros((1,pan.shape[2] ))),axis=0)
        zfla = panz.flatten('f')
        peakz=sgn.find_peaks(zfla, height=0, distance=1)
        zindex=np.asarray(peakz[0])
        # zindex2=zindex
        # for k in range(zindex.shape):
        zindex = (zindex) // (panz.shape[0]) + ((zindex) - (zindex) // (panz.shape[0]) * (panz.shape[0])) * pany.shape[1]
        if ((np.intersect1d(yindex,zindex)).shape[0]!=0):
            peak = (np.intersect1d(yindex,zindex))
            for k in range(peak.shape[0]):
                y=peak[k]// pany.shape[1]
                z = (peak[k]-peak[k] //  pany.shape[1]*pany.shape[1])
                center = pan[i, int(y), int(z)]
                if int(y) == 0 or int(y) == pan.shape[1] - 1 or int(z) == 0 or int(z) == pan.shape[2] - 1:
                    ax1=ax
                    if y == 0:
                        ax1=ax1[np.argwhere(ax1[:,1]>-1)[:,0],:]
                    if y == pan.shape[1] - 1:
                        ax1=ax1[np.argwhere(ax1[:,1]<1)[:,0],:]
                    if z == 0:
                        ax1=ax1[np.argwhere(ax1[:,2]>-1)[:,0],:]
                    if z == pan.shape[2] - 1:
                        ax1=ax1[np.argwhere(ax1[:,2]<1)[:,0],:]
                    squ = np.zeros(ax1.shape[0])
                    for h in range(ax1.shape[0]):
                        squ[h] = pan[i + ax1[h, 0], int(int(y) + ax1[h, 1]), int(int(z) + ax1[h, 2])]
                    if squ.all() < center:
                        if center > panmax *rate:
                            kk.append([i, int(y), int(z)])
                    continue

                squ=np.zeros(26)
                # print()
                for h in range(ax.shape[0]):
                    squ[h]=pan[i+ax[h,0],int(int(y)+ax[h,1]),int(int(z)+ax[h,2])]
                if (squ<center ).all():
                    if center > panmax*rate:
                        kk.append([i,int(y),int(z)])
#

    if savenpy==1:
        np.savez(os.path.join(saveDirPath, semName + suffix),
                 semArray=semArray,
                 estimationLocation=positionArray[xLocationIndex, yLocationIndex, zLocationIndex],
                 maxSemLocation=np.array([xLocationIndex, yLocationIndex, zLocationIndex]),
                 maxSemIndex=np.max(semArray),
                 refTraceindex=[refIndexP, refIndexS],
                 refPoint=[refPointP, refPointS],
                 app=apDataArrayP,
                 aps=apDataArrayS,
                 appa=appa,
                 picknum=appa.shape[0]
                 )
    posp = positionArray[xLocationIndexP, yLocationIndexP, zLocationIndexP][0, :]
    poss = positionArray[xLocationIndexS, yLocationIndexS, zLocationIndexS][0, :]
    txt = np.concatenate((positionArray[xLocationIndex, yLocationIndex, zLocationIndex][0, :],
                          np.array([np.max(semArray)]),np.array([np.min(semArray)]),np.asarray([len(kk),num])),
                         axis=0)
    semmax=np.max(semArray)
    return txt,semmax,len(kk),num

