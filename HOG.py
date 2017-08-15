import cv2
import numpy as np
import time
import operator
import math
import glob
import pickle
import functools
from os import path
from sklearn.svm import *
from sklearn.externals import joblib
def categorize(angles,magnitudes):
    HistogramBins = np.zeros((9,1), dtype=np.float32)
    angles=np.degrees(angles)
    angles[angles>=180]=angles[angles>=180]-180
    bins=(angles.astype(np.int32)/20).ravel()
    magnitudes=magnitudes.ravel()
    HistogramBins=np.bincount(bins,magnitudes.ravel(),9)
    #for i in range(len(bins)):
        #HistogramBins[bins[i]-1]+=magnitudes[i]
    return HistogramBins
def getGradients(I):
    start=time.time()
    Ix=cv2.Sobel(I,5,1,0).astype(np.float32)
    Iy=cv2.Sobel(I,5,0,1).astype(np.float32)
    Ix,Iy=np.vsplit(Ix,10),np.vsplit(Iy,10)
    Ix,Iy=map(lambda Ix:np.hsplit(Ix,10),Ix),map(lambda Iy:np.hsplit(Iy,10),Iy)
    magnitude,angle=[],[]

    for i in range(len(Ix)):
            mag=map(cv2.cartToPolar,Ix[i],Iy[i])
            magnitude.append(map(operator.itemgetter(0),mag))
            angle.append(map(operator.itemgetter(1),mag[:]))

    return magnitude,angle
def getHOG(magnitude,angle):
    HOG=[]
    for i in range(len(angle)):
        for j in range(len(angle[i])):
            HOG.append(categorize(angle[i][j],magnitude[i][j]))
    return HOG
svm=[]
def classfiy(gray):
    global svm
    if(svm==[]):
        svm=joblib.load('HOGSVM.pkl')
    equalized = cv2.equalizeHist(gray)
    magnitude, angle = getGradients(equalized)
    HOG = getHOG(magnitude, angle)
    result = svm.predict(np.array(HOG).ravel())
    return result
def main():

    if(path.exists('HOGSVM.pkl')):
        svm=joblib.load('HOGSVM.pkl')
    else:
        print "here"
        TrafficSigns=glob.glob('/Users/apple/Downloads/Test Drive/DataSet/ts*.png')
        NonTrafficSigns=glob.glob('/Users/apple/Downloads/Test Drive/DataSet/nonts*.png')
        Labels=[]
        SVMsamples=[]
        for file in TrafficSigns:
            I=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            I=cv2.equalizeHist(I)
            magnitude,angle=getGradients(I)
            HOG=getHOG(magnitude,angle)
            SVMsamples.append(np.array(HOG).ravel())
            Labels.append(1.0)
        for file in NonTrafficSigns:
            I=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            I = cv2.equalizeHist(I)
            magnitude, angle = getGradients(I)
            HOG = getHOG(magnitude, angle)
            SVMsamples.append(np.array(HOG).ravel())
            Labels.append(-1.0)
        svm=SVC(kernel='linear',tol=0.00001,max_iter=-1)
        svm.fit(np.array(SVMsamples),np.array(Labels))
        joblib.dump(svm,'HOGSVM.pkl')
    start=time.time()
    test=cv2.imread('/Users/apple/Downloads/Test Drive/whatever/1.png',cv2.IMREAD_GRAYSCALE)
    test=cv2.equalizeHist(test)
    magnitude,angle=getGradients(test)
    HOG=getHOG(magnitude,angle)
    result=svm.predict(np.array(HOG).ravel())
    finish=float(time.time())
    print 'result is %d'%result
    print "time is %f" % (finish-start)
if __name__ == '__main__':
    main()
