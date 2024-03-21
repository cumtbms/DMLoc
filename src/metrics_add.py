from keras import backend as K
import scipy.signal as sgn
import numpy as np

def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
     The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
     This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
     With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
     # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
     Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def pick_analysis(predict , label , pWindowLen = 80 , sWindowLen = 80 ,  minDistance= 21 ,rReturn = False):
    """
    predict:  (3000,3)
    label: [itp its]

    :return:
    """

    waveLen = predict.shape[0]

    itp = label[0]
    its = label[1]




    pPickErrorNum = 0
    pPickSuccess = 0
    pLeakNum = 0


    sPickErrorNum = 0
    sPickSuccess = 0
    sLeakNum = 0



    #p以一个矩形窗滑动，找峰值
    pPredict = predict[:,0]
    pWindow = np.ones(pWindowLen)
    padLen =  int((pWindowLen + 1)/2)
    pPredict = np.pad(pPredict , ( padLen , padLen) , 'constant' ,constant_values=(0,0))

    pRes = np.zeros(waveLen)
    for i in range(waveLen):
        pRes[i] = np.sum(pWindow * pPredict[ i : i+pWindowLen])/pWindowLen

    pPeakArray = sgn.find_peaks( pRes , height=0.2 , distance= minDistance )
    pPeakArraySize = pPeakArray[0].shape[0]


    if  pPeakArraySize == 0:
         if itp >= 0 and itp < waveLen:
            # p漏拾
            pLeakNum = 1
    else:
        for i in range(pPeakArraySize):
            # print('***:', abs(pPeakArray[i] - itp))
            if abs(pPeakArray[0][i] - itp) > pWindowLen:
                pPickErrorNum = pPickErrorNum + 1
            else:
                pPickSuccess = 1

        if pPickSuccess == 0:
            pLeakNum = 1




    # s以一个矩形窗滑动，找峰值
    sPredict = predict[:, 1]
    sWindow = np.ones(sWindowLen)
    padLen = int((sWindowLen + 1) / 2)
    sPredict = np.pad(sPredict, (padLen, padLen), 'constant', constant_values=(0, 0))

    sRes = np.zeros(waveLen)
    for i in range(waveLen):
        sRes[i] = np.sum(sWindow * sPredict[ i : i+sWindowLen]) / sWindowLen

    sPeakArray = sgn.find_peaks(sRes, height=0.2, distance= minDistance  )
    sPeakArraySize = sPeakArray[0].shape[0]

    if  sPeakArraySize == 0:
         if its >= 0 and its < waveLen:
            # p漏拾
            sLeakNum = 1
    else:
        for i in range(sPeakArraySize):
            # print('***:',abs(sPeakArray[i] - its))
            if abs(sPeakArray[0][i] - its) > sWindowLen:
                sPickErrorNum = sPickErrorNum + 1
            else:
                sPickSuccess = 1

        if sPickSuccess == 0:
            sLeakNum = 1






    if not rReturn:
        return [pPickSuccess , pPickErrorNum , pLeakNum] , [sPickSuccess ,sPickErrorNum , sLeakNum]
    else:
        return  [pPickSuccess , pPickErrorNum , pLeakNum] , [sPickSuccess ,sPickErrorNum , sLeakNum], pRes ,sRes ,pPeakArray[0] , sPeakArray[0]





# def pick_analysis_batch( batch_predict , batch_itp , batch_its  ):
def pick_analysis_batch( y_true,y_pre):

    batch_p_array = np.array([0,0,0])
    batch_s_array = np.array([0,0,0])

    batch_size = y_pre.shape[0]
    print(y_pre.shape)
    for i in range(batch_size):
        per_p , per_s = pick_analysis( y_pre[i] ,  y_true[i] )
        batch_p_array = batch_p_array + np.array(per_p)
        batch_s_array = batch_s_array + np.array(per_s)
    return batch_p_array ,batch_s_array

def precall(y_true,y_pre):
    pArray,_=pick_analysis_batch(y_true, y_pre)
    batch_size = y_pre.shape[0]
    p_recall =   pArray[0]/(pArray[0]+ pArray[2]+ 0.001)
    return p_recall

def paccuracy(y_true,y_pre):
    pArray,_=pick_analysis_batch(y_true, y_pre)
    batch_size = y_pre.shape[0]
    p_accuracy = pArray[0]/(  batch_size +  0.001)
    return p_accuracy

def srecall(y_true,y_pre):
    _,sArray=pick_analysis_batch(y_true, y_pre)
    batch_size = y_pre.shape[0]
    s_recall =   sArray[0]/(sArray[0]+ sArray[2]+ 0.001)
    return s_recall

def saccuracy(y_true,y_pre):
    _,sArray=pick_analysis_batch(y_true, y_pre)
    batch_size = y_pre.shape[0]
    s_accuracy = sArray[0]/(  batch_size +  0.001)
    return s_accuracy