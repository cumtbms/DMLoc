import numpy as np
from raytracingCUDA import  ratracingcuda
from numba import cuda
import copy
import time
def shot_method_VTIGPUnodict(rcvPath, velPath, delta = 10, timeSavePath = None,
                             startPosition = None ,sourceNum = None,**kwargs):  # phaseMode 0  P ；  1 S
    starttime = time.time()
    accuracy = None if "accuracy" not in kwargs.keys() else kwargs.get("accuracy")
    maxIter = None if "maxIter" not in kwargs.keys() else kwargs.get("maxIter")
    rcv = np.loadtxt(open(rcvPath,'rb'),delimiter=',',skiprows=1,usecols=[0,2,3,4])
    rcvNameList = [ str(int(rcv[i,0])) for i in range(rcv.shape[0])]
    Rec_ver = rcv[:,1:].astype(np.float32)
    Rec_Num = Rec_ver.shape[0]

    sourceNum = np.asarray(sourceNum)
    deltX = delta
    deltY = delta
    deltZ = delta
    xNum = sourceNum[0]
    yNum = sourceNum[1]
    zNum = sourceNum[2]
    aStart_pos = np.asarray(startPosition)
    timesp = np.zeros((sourceNum[0] * sourceNum[1] * sourceNum[2], Rec_Num))
    timess = np.zeros((sourceNum[0] * sourceNum[1] * sourceNum[2], Rec_Num))
    source = np.zeros((sourceNum[0] * sourceNum[1] * sourceNum[2], 3))
    srcnum = sourceNum[0] * sourceNum[1] * sourceNum[2]

    vel = np.loadtxt(open(velPath, 'rb'), delimiter=',', skiprows=1, usecols=[0,1, 2])
    pVelVector, sVelVector, layerVector = vel[:,1].astype(np.float32),vel[:,2].astype(np.float32),vel[:,0].astype(np.float32)
    # aLayerz, aVel
    Accuracy = -1
    Max_iter = -1
    threads_per_block = 64
    blocks_per_grid_x = (xNum * yNum * zNum + threads_per_block - 1) // threads_per_block
    blocks_per_grid_y = Rec_Num
    griddim = (blocks_per_grid_x, blocks_per_grid_y)
    blockdim = (threads_per_block, 1)
    cutimesp = cuda.to_device(timesp)
    cutimess = cuda.to_device(timess)
    cusource = cuda.to_device(source)
    cuRec_ver = cuda.to_device(Rec_ver)
    cuaStart_pos = cuda.to_device(aStart_pos)
    ratracingcuda[griddim, blockdim](xNum, yNum, zNum, cuaStart_pos, deltX, deltY, deltZ, cuRec_ver, \
                                     layerVector, pVelVector, cutimesp, cusource, Accuracy, Max_iter, srcnum)
    cuda.synchronize()
    gputimep = cutimesp.copy_to_host()

    ratracingcuda[griddim, blockdim](xNum, yNum, zNum, cuaStart_pos, deltX, deltY, deltZ, cuRec_ver, \
                                     layerVector, sVelVector, cutimess, cusource, Accuracy, Max_iter, srcnum)
    cuda.synchronize()
    gputimes = cutimess.copy_to_host()
    np.savez(timeSavePath,ptime=gputimep,stime=gputimes,
             rcvNameList=rcvNameList,delta=delta,startPosition=startPosition,
             sourcenum=sourceNum)


if __name__ == "__main__":

    starttime=time.time()
    shot_method_VTIGPUnodict(rcvPath=r'./bcrcv.csv',
                             velPath=r'./bcvel.csv',
                             timeSavePath='./wtt',
                             delta=100,
                             startPosition=[0,0,-4000],#[5400,5610,-2000]
                             sourceNum=[300,360,30],                       
                             )
    print(time.time()-starttime)