import time
import scipy.signal as sgn
from scipy.signal import find_peaks
from numba import njit,jit
import numba
import numpy as np
from numba import cuda

@cuda.jit
def getres(res,predict,WindowLen,colnum,rownum):
    row= cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    col= cuda.blockIdx.x + cuda.blockIdx.y*cuda.gridDim.x
    if col <colnum and row<rownum:
        for i in range(WindowLen):
            res[row, col]=res[row, col]+predict[row,col+i]
        res[row, col]=res[row, col]/WindowLen