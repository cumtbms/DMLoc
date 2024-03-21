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
        # sxyz_x = custartposition[0] + deltx / 2 + xno * deltx
        # sxyz_y = custartposition[1] + delty / 2 + yno * delty
        # sxyz_z = custartposition[2] + deltz / 2 + zno * deltz
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
        #times[sourcethrxno + recno * Src_Num, 0] = Travel_time
        # if sourcethrxno == 29184 and recno==1:
        #     print(Hor_distance,T_distance,sxyz_x,sxyz_y,sxyz_z,Src_Num,recno,xno,yno,zno,custartposition[0],Top_point,Bottom_point,Travel_time)
    # if index == 82944 :
    #     print(Hor_distance, T_distance, sxyz_x, sxyz_y, sxyz_z, Src_Num, recno, Travel_time)





def main():
    # d1 = 'rec5.mat'
    # dd1 = sio.loadmat(d1)
    # RecX = dd1['rex']
    # RecY = dd1['rey']
    # RecZ = dd1['rez']
    # Rec_Num = RecZ.shape[1]
    # Rec_ver = np.zeros([Rec_Num, 3])
    # start_depth = 1488
    # Rec_ver[:, 0] = RecX[0, :]
    # Rec_ver[:, 1] = RecY[0, :]
    # Rec_ver[:, 2] = RecZ[0, :]-float(start_depth)
    ##################
    # rec =[[-1081.0, 647.0, -133.21], [-839.0, 1112.0, -200.11], [-498.0, 1548.0, -155.56], [70.0, 1523.0, -102.28],
    #  [173.0, 1124.0, -152.18], [-609.0, 406.0, -237.92], [-751.0, -151.0, -243.73], [-1214.0, -773.0, -183.22],
    #  [-474.0, -641.0, -264.57], [-135.0, -82.0, -233.73], [132.0, 352.0, -221.36], [300.0, 817.0, -177.2],
    #  [470.0, 1096.0, -207.86], [760.0, 1654.0, -87.73], [1188.0, 981.0, -280.69], [1295.0, 242.0, -90.6],
    #  [681.0, -104.0, -211.13], [439.0, -569.0, -232.18], [1177.0, -314.0, -141.81], [-722.0, 1699.0, -145.58],
    #  [-990.0, 1327.0, -187.64], [-952.0, 94.0, -258.56], [-1054.0, 432.0, -228.59], [490.0, 1528.0, -51.34],
    #  [-92.0, 566.0, -206.96], [70.0, -728.0, -213.93], [871.0, 545.0, -282.84], [794.0, 853.0, -240.15],
    #  [1184.0, 1258.0, -278.16], [700.0, 389.0, -277.66], [358.0, 16.0, -276.12], [104.0, 722.0, -271.96],
    #  [2.0, 937.0, -254.79], [-248.0, 1273.0, -108.18], [-442.0, 901.0, -121.35], [615.0, 1406.0, -145.55],
    #  [839.0, 1254.0, -132.42], [625.0, 450.0, -365.0], [625.0, 450.0, -415.0], [625.0, 450.0, -465.0],
    #  [625.0, 450.0, -515.0], [625.0, 450.0, -565.0], [625.0, 450.0, -615.0], [625.0, 450.0, -665.0],
    #  [625.0, 450.0, -715.0], [625.0, 450.0, -765.0], [625.0, 450.0, -815.0], [625.0, 450.0, -865.0],
    #  [625.0, 450.0, -915.0], [625.0, 450.0, -965.0], [625.0, 450.0, -1015.0], [625.0, 450.0, -1065.0],
    #  [625.0, 450.0, -1115.0], [625.0, 450.0, -1165.0], [625.0, 450.0, -1215.0], [625.0, 450.0, -1265.0],
    #  [625.0, 450.0, -1315.0], [625.0, 450.0, -1365.0], [625.0, 450.0, -1415.0], [625.0, 450.0, -1465.0],
    #  [625.0, 450.0, -1515.0], [625.0, 450.0, -1565.0], [625.0, 450.0, -1615.0], [625.0, 450.0, -1665.0]]
    #地面
    rec =[[-1081.0, 647.0, -133.21], [-839.0, 1112.0, -200.11], [-498.0, 1548.0, -155.56], [70.0, 1523.0, -102.28],
     [173.0, 1124.0, -152.18], [-609.0, 406.0, -237.92], [-751.0, -151.0, -243.73], [-1214.0, -773.0, -183.22],
     [-474.0, -641.0, -264.57], [-135.0, -82.0, -233.73], [132.0, 352.0, -221.36], [300.0, 817.0, -177.2],
     [470.0, 1096.0, -207.86], [760.0, 1654.0, -87.73], [1188.0, 981.0, -280.69], [1295.0, 242.0, -90.6],
     [681.0, -104.0, -211.13], [439.0, -569.0, -232.18], [1177.0, -314.0, -141.81], [-722.0, 1699.0, -145.58],
     [-990.0, 1327.0, -187.64], [-952.0, 94.0, -258.56], [-1054.0, 432.0, -228.59], [490.0, 1528.0, -51.34],
     [-92.0, 566.0, -206.96], [70.0, -728.0, -213.93], [871.0, 545.0, -282.84], [794.0, 853.0, -240.15],
     [1184.0, 1258.0, -278.16], [700.0, 389.0, -277.66], [358.0, 16.0, -276.12], [104.0, 722.0, -271.96],
     [2.0, 937.0, -254.79], [-248.0, 1273.0, -108.18], [-442.0, 901.0, -121.35], [615.0, 1406.0, -145.55],
     [839.0, 1254.0, -132.42]]
    #井中
    # rec =[ [625.0, 450.0, -365.0], [625.0, 450.0, -415.0], [625.0, 450.0, -465.0],
    #  [625.0, 450.0, -515.0], [625.0, 450.0, -565.0], [625.0, 450.0, -615.0], [625.0, 450.0, -665.0],
    #  [625.0, 450.0, -715.0], [625.0, 450.0, -765.0], [625.0, 450.0, -815.0], [625.0, 450.0, -865.0],
    #  [625.0, 450.0, -915.0], [625.0, 450.0, -965.0], [625.0, 450.0, -1015.0], [625.0, 450.0, -1065.0],
    #  [625.0, 450.0, -1115.0], [625.0, 450.0, -1165.0], [625.0, 450.0, -1215.0], [625.0, 450.0, -1265.0],
    #  [625.0, 450.0, -1315.0], [625.0, 450.0, -1365.0], [625.0, 450.0, -1415.0], [625.0, 450.0, -1465.0],
    #  [625.0, 450.0, -1515.0], [625.0, 450.0, -1565.0], [625.0, 450.0, -1615.0], [625.0, 450.0, -1665.0]]
    Rec_ver = np.array(rec)
    Rec_Num = Rec_ver.shape[0]


    deltX = 10
    deltY = 10
    deltZ = 10
    # deltX = 20
    # deltY = 20
    # deltZ = 20
    # Vel = [2665.1, 2665.1, 3642.3, 3703.7, 3455.29, 3658.54, 3448.28, 3750, 4006.97, 4245.28, 4504]
    # aVel = np.asarray(Vel)
    # Layerz = [1600 - 1600, 820 - 1600, 740 - 1600, 580 - 1600, 390 - 1600, 120 - 1600, 70 - 1600, -10 - 1600,
    #           -250 - 1600, -370 - 1600, -1010 - 1600]
    # aLayerz = np.asarray(Layerz)
    ########################
    Vel =[2665.1, 3642.3, 3703.7, 3455.29, 3658.54, 3448.28, 3750.0, 4006.97, 4245.28, 4504.0, 4504.0]#p_vel
    aVel = np.asarray(Vel)
    Layerz = [0.0, -380.0, -460.0, -620.0, -810.0, -1080.0, -1130.0, -1210.0, -1450.0, -1570.0, -2498.0]
    aLayerz = np.asarray(Layerz)

    #print(aLayerz)
    # 40205 20000 5
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596282, 4105690, -100 - start_depth]
    # 50405 50000 5/50405 100000 10
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596382, 4105890, -100 - start_depth]
    # 80505 100000 5/80505 200000 10
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596682, 4105990, -100 - start_depth]

    # 100805 200000 5
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596882, 4106290, -100 - start_depth]
    # 20205 20000 10
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596082, 4105690, -100 - start_depth]
    # 40255 50000 10
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [596282, 4105740, -100 - start_depth]
    # 2002005 500000 5
    # Start_pos = [595882, 4105490, -150 - start_depth]
    # End_pos = [597882, 4107490, -100 - start_depth]
##########################
    Start_pos = [-470, - 740, - 1685]
    Start_pos = [-470, - 740, - 2185]
    aStart_pos = np.asarray(Start_pos)
    xNum = 100
    yNum = 200
    zNum = 10
    # xNum = 50
    # yNum = 100
    # zNum = 50
    # xNum = int((End_pos[0] - Start_pos[0]) // deltX)
    # yNum = int((End_pos[1] - Start_pos[1]) // deltX)
    # zNum = int((End_pos[2] - Start_pos[2]) // deltX)
    srcnum = xNum * yNum * zNum
    times = np.zeros((xNum * yNum * zNum,Rec_Num))
    source = np.zeros((xNum * yNum * zNum, 3))
    Accuracy = -1
    Max_iter = -1
    #gpu
    threads_per_block = 64
    blocks_per_grid_x=(xNum * yNum * zNum +threads_per_block-1)// threads_per_block
    blocks_per_grid_y=Rec_Num
    griddim=(blocks_per_grid_x,blocks_per_grid_y)
    blockdim=(threads_per_block,1)
    cutimes=cuda.to_device(times)
    cusource=cuda.to_device(source)
    cuRec_ver=cuda.to_device(Rec_ver)
    cuaStart_pos=cuda.to_device(aStart_pos)
    processtime=time()
    fuzhi[griddim, blockdim](xNum ,yNum ,zNum , cuaStart_pos, deltX, deltY, deltZ, cuRec_ver,\
                                              aLayerz, aVel, cutimes, cusource, Accuracy, Max_iter,srcnum)
    cuda.synchronize()
    outtimes=cutimes.copy_to_host()
    outsource=cusource.copy_to_host()
    sio.savemat('dimiangpu200000-1.mat', {'arrival_times': outtimes, 'src_pos': outsource})
    print(processtime-time())

if __name__ == "__main__":
    main()