from numpy.lib.scimath import sqrt
import numpy as np
from numpy import array, linspace, ones, zeros, empty, repeat, \
    transpose, diff, where, real, cumsum, append, multiply, \
    arcsin, finfo, concatenate, square, flipud
from numba import cuda
from time import time
import math
import scipy.io as sio

@cuda.jit
def ratracingcuda(X_Num ,Y_Num ,Z_Num,custartposition,deltx,delty,deltz,Rec_ver,Layerz,Vel,times, source,Accuracy, Max_iter,Src_Num):
    # index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x +cuda.blockIdx.y * cuda.blockDim.x*cuda.gridDim.x
    sourcethrxno= cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    restindex = (cuda.blockDim.x * cuda.gridDim.x - Src_Num) * cuda.blockIdx.y
    recno = cuda.blockIdx.y
    index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x +cuda.blockIdx.y * cuda.blockDim.x*cuda.gridDim.x
    if sourcethrxno < Src_Num:
        zno =(index - Src_Num * recno-restindex) // (X_Num * Y_Num)
        yno = ((index - Src_Num * recno - (X_Num * Y_Num) * zno-restindex) // X_Num)
        xno = index - Src_Num * recno - (X_Num * Y_Num) * zno - X_Num * yno-restindex        
        sxyz_x = custartposition[0] + xno * deltx
        sxyz_y = custartposition[1] + yno * delty
        sxyz_z = custartposition[2] + zno * deltz
        Hor_distance = math.sqrt(pow((sxyz_x - Rec_ver[recno, 0]), 2) + pow((sxyz_y - Rec_ver[recno, 1]), 2))
        T_distance = math.sqrt(pow((sxyz_x - Rec_ver[recno, 0]), 2) + pow((sxyz_y - Rec_ver[recno, 1]), 2) + \
                               pow((sxyz_z - Rec_ver[recno, 2]), 2))
        if recno==0:
            source[sourcethrxno,0]=sxyz_x
            source[sourcethrxno,1]=sxyz_y
            source[sourcethrxno,2]=sxyz_z
        if Max_iter == -1:
            Max_iter = 10000
        if Accuracy == -1:
            Accuracy = 0.0001
        #recz=Rec_ver[recno, 2]
        #print(cuRec_ver[recno,2],cuRec_ver[recno, 1],cuRec_ver[recno, 0])
        if sxyz_z>Rec_ver[recno, 2]:
            Top_point = sxyz_z  # 深度以负数输入，Top表示是最上层
            Bottom_point =Rec_ver[recno, 2]
            # print(Top_point,Rec_ver[recno,2],recno,sxyz_x,sxyz_y,sxyz_z)
        else:
            Top_point = Rec_ver[recno, 2] # 深度以负数输入，Top表示是最上层
            Bottom_point = sxyz_z
            # print(Top_point,Rec_ver[recno,2],recno,sxyz_x,sxyz_y,sxyz_z)
        forLnum = 0
        TLidx = -1
        Layerlen = int(len(Layerz))
        for i in range(Layerlen):
            t=int(Layerz[i])
            if Top_point > t and Bottom_point < t :
                forLnum = forLnum + 1
                if TLidx == -1:
                    TLidx = i
        if forLnum == 0:
            for i in range(Layerlen):
                t = int(Layerz[i])
                if Top_point >= t :
                    if TLidx == -1:
                        TLidx = i-1
            Travel_time=T_distance/Vel[TLidx]
        else:
            BLidx = TLidx + forLnum
            Thickidx = TLidx
            velmax = Vel[TLidx - 1]
            Velsum=Vel[TLidx - 1]
            for i in range(forLnum):
                Thickidx = TLidx + i
                Velsum = Velsum + Vel[Thickidx]
                if Vel[Thickidx] > velmax:
                    velmax = Vel[Thickidx]
            Velmean=Velsum/(forLnum+1)
            Ray_parameter_small = 0
            Ray_parameter_large = 1 / velmax
            Alpha_small_sin = Vel[TLidx - 1] * Ray_parameter_small
            Alpha_small = math.asin(Alpha_small_sin)
            Alpha_large_sin = Vel[TLidx - 1] * Ray_parameter_large
            Alpha_large = math.asin(Alpha_large_sin)
            tan_alpha = math.tan(Alpha_small)
            Horizen_small = (Top_point - Layerz[TLidx]) * tan_alpha
            tan_alpha = math.tan(Alpha_large)
            Horizen_large = (Top_point - Layerz[TLidx]) * tan_alpha
            for i in range(forLnum):
                Thickidx = TLidx + i
                Alpha_small_sin = Vel[Thickidx] * Ray_parameter_small
                Alpha_small = math.asin(Alpha_small_sin)
                tan_alpha_small = math.tan(Alpha_small)
                Horizen_small = Horizen_small + (Layerz[Thickidx] - Layerz[Thickidx + 1]) * tan_alpha_small
                Alpha_large_sin = Vel[Thickidx] * Ray_parameter_large
                Alpha_large = math.asin(Alpha_large_sin)
                tan_alpha_large = math.tan(Alpha_large)
                Horizen_large = Horizen_large + (Layerz[Thickidx] - Layerz[Thickidx + 1]) * tan_alpha_large
            Horizen_small = Horizen_small - (Bottom_point - Layerz[BLidx]) * tan_alpha_small
            Horizen_large = Horizen_large - (Bottom_point - Layerz[BLidx]) * tan_alpha_large
            Z_difference = Top_point - Bottom_point
            if (abs(Z_difference) < Accuracy):
                Travel_time = math.sqrt(pow(Hor_distance , 2) + pow(T_distance ,2)) /Velmean
                Ray_parameter = Ray_parameter_large
            else:
                Itteration = 0

                while ((abs(Horizen_small - Horizen_large) > Accuracy) & (Itteration < Max_iter)):
                    Itteration = Itteration + 1
                    Ray_parameter = (Ray_parameter_large + Ray_parameter_small) / 2
                    Alpha_sin = Vel[TLidx - 1] * Ray_parameter
                    Alpha = math.asin(Alpha_sin)
                    tan_alpha = math.tan(Alpha)
                    Horizen = (Top_point - Layerz[TLidx]) * tan_alpha
                    for i in range(forLnum):
                        Thickidx = TLidx + i
                        Alpha_sin = Vel[Thickidx] * Ray_parameter
                        Alpha = math.asin(Alpha_sin)
                        tan_alpha = math.tan(Alpha)
                        Horizen = Horizen + (Layerz[Thickidx] - Layerz[Thickidx + 1]) * tan_alpha
                    Horizen = Horizen - (Bottom_point - Layerz[BLidx]) * tan_alpha
                    if (Horizen > Hor_distance):
                        Ray_parameter_large = Ray_parameter
                        Horizen_large = Horizen
                    else:
                        Ray_parameter_small = Ray_parameter
                        Horizen_small = Horizen
                if (abs(Horizen_small - Horizen_large) > Accuracy):
                    print('accuracy cannot be achieved', Itteration, Hor_distance, T_distance, sxyz_x, sxyz_y, sxyz_z, xno, yno, zno, Top_point, Bottom_point, Rec_ver[recno, 2], recno)
                    #print('accuracy cannot be achieved', Itteration)

                Ray_parameter = (Ray_parameter_large + Ray_parameter_small) / 2
                Alpha_sin = Vel[TLidx - 1] * Ray_parameter
                Alpha = math.asin(Alpha_sin)
                cos_alpha = math.cos(Alpha)
                Travel_time = (Top_point - Layerz[TLidx]) /( cos_alpha * Vel[TLidx - 1])
                for i in range(forLnum):
                    Thickidx = TLidx + i
                    Alpha_sin = Vel[Thickidx] * Ray_parameter
                    Alpha = math.asin(Alpha_sin)
                    cos_alpha = math.cos(Alpha)
                    Travel_time = Travel_time + (Layerz[Thickidx] - Layerz[Thickidx + 1]) / (cos_alpha * Vel[Thickidx])
                Travel_time = Travel_time - (Bottom_point - Layerz[BLidx]) / (cos_alpha * Vel[Thickidx])
        times[sourcethrxno, recno]=Travel_time
       
