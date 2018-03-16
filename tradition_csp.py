import scipy.io as scio
s=scio.loadmat(r'E:\python\csp_r集合集成学习\BCICIV_calib_ds1g.mat')
ss=s['cnt']*0.1
ss=ss.T
y=s['mrk']['y'][0][0][0]
pos=s['mrk']['pos'][0][0][0]
from scipy.signal import butter,lfilter
import numpy as np
sample=np.zeros((200,59,200))
for i in range(200):
    for j in range(59):
        sample[i][j][:]=ss[j][pos[i]+200:pos[i]+400]
b,a=butter(5,[8/50,30/50],btype='bandpass')
samplefilted=np.ones((200,59,200))
for i in range(200):
    for j in range(59):
        samplefilted[i][j][:]=lfilter(b,a,sample[i][j][:])
class1=np.zeros((100,59,200))
class2=np.zeros((100,59,200))
count1=0
count2=0
for i in range(200):
    if y[i]>0:
        class1[count1][:][:]=samplefilted[i][:][:]
        count1+=1
    else:
        class2[count2][:][:]=samplefilted[i][:][:]
        count2+=1
a=range(100)
b1=np.reshape(a,(10,10))
acc=np.zeros((10))
for k in range(10):
    b2=np.setdiff1d(a,b1[k])
    class1_train=class1[b2]
    class1_test=class1[b1[k]]
    class2_train=class2[b2]
    class2_test=class2[b1[k]]
    R1=np.zeros((59,59))
    R2=np.zeros((59,59))
    for i in range(90):
        R1=R1+np.cov(class1_train[i])
        R2=R2+np.cov(class2_train[i])
    R1=R1/90
    R2=R2/90
    R3=R1+R2
    Sigma, U0 = np.linalg.eig(R3)
    P = np.dot(np.diag(Sigma ** (-0.5)), U0.T)
    YL = np.dot(np.dot(P, R1), P.T)
    SigmaL, UL = np.linalg.eig(YL)
    I = np.argsort(SigmaL)
    F1 = np.dot(UL.T, P)[[0, 1, 57, 58], :]
    featuref1=np.zeros((90,4))
    featuref2=np.zeros((90,4))
    tes1=np.zeros((10,4))
    tes2=np.zeros((10,4))
    for i in range(90):
        dataf11=np.dot(class1_train[i].T,F1.T)
        dataf12=np.dot(class2_train[i].T,F1.T)
        for j in range(4):
            featuref1[i,j]=np.log(np.var(dataf11[:,j]))
            featuref2[i,j]=np.log(np.var(dataf12[:,j]))
    traindata=np.vstack((featuref1,featuref2))
    for i in range(10):
        dataf11=np.dot(class1_test[i].T,F1.T)
        dataf12=np.dot(class2_test[i].T,F1.T)
        for j in range(4):
            tes1[i,j]=np.log(np.var(dataf11[:,j]))
            tes2[i,j]=np.log(np.var(dataf12[:,j]))
    testdata=np.vstack((tes1,tes2))
    from sklearn.svm import SVC
    svm=SVC(kernel='linear',C=1.0,random_state=0)
    svm.fit(traindata,np.vstack((np.ones((90,1)),np.ones((90,1))*-1)))
    predict_label=svm.predict(testdata)
    acc[k]=np.sum(predict_label==np.squeeze(np.vstack((np.ones((10,1)),np.ones((10,1))*-1))))/20
print(np.mean(acc))