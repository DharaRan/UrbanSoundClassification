# -*- coding: utf-8 -*-
"""
Pearson Correlation 
"""

import math
import numpy as np
import pickle

def load_data(filename):
    print('loading %s...' % filename)
    return pickle.load(open(filename, "rb"), encoding="latin1")

  
def inv_one_hot_encode(labels):  # inverse (index of 1 for each row)
    return np.argmax(labels, axis=1)

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


def selectTopBestR(data,label,top):
    cols=len(data[0])
    rValues=[]
    for c in range(0,cols):
        column=data[:,c]
        val=abs(pearson_def(column, label))
        rValues.append(val)
        indices = sorted(range(len(rValues)), key=rValues.__getitem__, reverse=True)# sorts the T list from greatest to least
        idx = indices[:top]# this will get the "top" number of indices
    
    d=data[:,idx]
    return(d)

def dump_data(path, objects, names, ext='.pkl'):
    print('dumping %s...' % names)
    pklfile=open(path + names + ext, "wb")
    pickle.dump(objects,pklfile)
    pklfile.close()

"""
#################################################################################
######################### Read in Data ##########################################
#################################################################################
"""


FOLDER = [1,2,3,4,5,6,7,8,9,10]


folderDir='data128norm/features_specsNorm_fold'
folderDirLabel='data128norm/labels_specsNorm_fold'
SAVE_DIR = 'C:\\Dhara\\Cs698_DataScienceTopics\\Project\\UrbanSound8K\\data139features_pearsonCorr\\'


"""
# READ data and label data
"""
for i in FOLDER:
    print('On Folder: ', str(i))
    filename=folderDir+str(i)+'.pkl'
    data=load_data(filename)
    
    Label_filename=folderDirLabel+str(i)+'.pkl'
    label=load_data(Label_filename)
      
    #print('Data shape: ',data.shape)
    #print('Label shape: ',label.shape)
    
    data=np.reshape(data,(len(data),len(data[0])*len(data[1])))
    label=inv_one_hot_encode(label)
    print('Getting Feature Selected Data...')
    newData=selectTopBestR(data,label,top=193)
       
    foldname='fold'+str(i)
    dump_data(SAVE_DIR, newData, foldname, ext='.pkl')  
    foldname='label'+str(i)
    dump_data(SAVE_DIR, label, foldname, ext='.pkl')    
    











