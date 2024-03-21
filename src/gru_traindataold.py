if __name__ == "__main__":
    import h5py
    from numpy.lib.scimath import sqrt
    import numpy as np
    from numpy import array, linspace, ones, zeros, empty, repeat, \
        transpose, diff, where, real, cumsum, append, multiply, \
        arcsin, finfo, concatenate, square, flipud
    import math
    import scipy.io as sio
    import random
    sample=100
    deltX = 100
    deltY = 100
    deltZ = 100
    Start_pos = [0,0,-4000]
    End_pos = [30000,36000,-1000]#[30000, 36000, -1000]
    aStart_pos = np.asarray(Start_pos)

    xNum = int((-Start_pos[0] + End_pos[0]) / deltX)
    yNum = int((-Start_pos[1] + End_pos[1]) / deltX)
    zNum = int((-Start_pos[2] + End_pos[2]) / deltX)
    rcvnorm=np.loadtxt(open(r"../data/bcrcv.csv", "rb"), delimiter=",", skiprows=1, usecols=[1,2,3]).T
    xmax=rcvnorm[1,:].max()-rcvnorm[1,:].min()
    ymax = rcvnorm[2, :].max()-rcvnorm[2,:].min()
    rcvnorm[1,:]=(rcvnorm[1,:]-rcvnorm[1,:].min())/xmax
    rcvnorm[2, :]=(rcvnorm[2,:]-rcvnorm[2,:].min())/ymax
    h=np.load('../data/wtt.npz')
    ptimes=h['ptime'][:]
    stimes=h['stime'][:]
    stanum=15
    h6 = h5py.File('grutraindata.h5', 'a')
    for i in range(xNum * yNum * zNum):
        phasep = np.zeros((1, rcvnorm.shape[1]))
        phases = np.ones((1, rcvnorm.shape[1]))
        envent = np.zeros((1, rcvnorm.shape[1]))
        tmin = ptimes[i:1 + i, :].min()
        move = np.random.randint(0, 1500, size=1) / sample
        concatp = np.concatenate((rcvnorm, ptimes[i:1 + i, :] - tmin+move, phasep,envent), axis=0)
        concats = np.concatenate((rcvnorm, stimes[i:1 + i, :] - tmin+move, phases,envent), axis=0)#id x y t phase envent
        sourcepos = str([i%xNum,i%(xNum * yNum)//xNum,i//(xNum * yNum)])
        # drop trace
        misstracenum=np.random.choice(int(stanum*0.15),1)[0]+1
        misstraceindex = np.random.choice(rcvnorm.shape[1],misstracenum, replace=False)
        for j in range(misstracenum):
            m = misstraceindex[j]
            concatp[:, m] = None
            concats[:, m] = None
        concats=concats[:, ~np.isnan(concats).any(axis=0)]
        concatp=concatp[:, ~np.isnan(concatp).any(axis=0)]
        # p-s s-p
        noisenum = np.random.choice(int(stanum*0.15),1)[0]+3
        noiseno = np.zeros(noisenum)
        noiseno[:noisenum//3]=1
        noiseindex = np.random.choice(rcvnorm.shape[1]-misstracenum, noisenum, replace=False)
        concatpcopy = np.array(list(concatp))
        concatscopy=np.array(list(concats))
        for k in range(noisenum):
            index = noiseindex[k]
            if noiseno[k] == 0:  # p-s
                concats=np.concatenate((concats,concatp[:,index:index+1]),axis=1)
                concats[4,-1]=1
                concats[5,-1]=1
                concatp[:,index]=None
            if noiseno[k] == 1:  # s-p
                concatp=np.concatenate((concatp,concats[:,index:index+1]),axis=1)
                concatp[4,-1]=0
                concatp[5,-1]=1
                concats[:,index]=None
        # add wrong/miss
        wmnum = np.random.choice(int(stanum*0.2), 1)[0] + 1  # number of wrong and miss
        wmindex = np.random.choice(rcvnorm.shape[1]-misstracenum,  wmnum, replace=False)
        wmno = np.random.choice(4, wmnum)
        for h in range(wmnum):
            index = wmindex[h]
            if wmno[h] == 0:  # wrongp
                wrongpt = np.random.randint(0, 600, size=1)[0] / 100
                while abs(wrongpt -concatpcopy[3,index]) < 0.2:
                    wrongpt = np.random.randint(0, 600, size=1)[0] / 100
                wrongp =np.asarray(list(concatpcopy[:,index:index+1]))
                wrongp[3,0]=wrongpt
                wrongp[5, 0] = 1
                concatp = np.concatenate((concatp, wrongp), axis=1)
            if wmno[h] == 1:  # wrongs
                wrongst = np.random.randint(0,  600, size=1)[0] / 500
                while abs(wrongst -concatscopy[3,index]) < 0.2:
                    wrongst = np.random.randint(0,  600, size=1)[0] / 500
                wrongs =np.asarray(list(concatscopy[:,index:index+1]))
                wrongs[3,0]=wrongst
                wrongs[5, 0] = 1
                concats = np.concatenate((concats, wrongs), axis=1)
            if wmno[h] == 2:  # missp
                concatp[:,index] = None
            if wmno[h] == 3:  # misss
                concats[:,index] = None
        concats=concats[:, ~np.isnan(concats).any(axis=0)].T
        concatp=concatp[:, ~np.isnan(concatp).any(axis=0)].T
        concatps = np.concatenate((concatp, concats), axis=0)#id x y t phase
        concatps = concatps[ np.argsort(concatps[:, 3]),:]
        if concatps.shape[0]<20:
            print(i,'小于12')
        concatps = concatps[:20,:]
        if np.isnan(concatps).any():
            print(i, 'nan')
        if i%10000==0:
            print(i,concatps.shape)
        h6.create_dataset(str(sourcepos), data=concatps)
    dt = -ptimes + stimes
    ptmin=np.min(ptimes,axis=1)
    stmax = np.max(stimes, axis=1)
    dmax=stmax-ptmin
    h6 = h5py.File('./grutraindata.h5', 'r')
    k=list(h6.keys())
    random.shuffle(k)
    f=open("grutraindata.txt","w")
    for line in k:
        f.write(line+'\n')
    f.close()