import scipy.io as scio
s=scio.loadmat(r'E:\python\csp_r集合集成学习\BCICIV_calib_ds1c.mat')
ss=s['cnt']*0.1
ss=ss.T
y=s['mrk']['y'][0][0][0]
pos=s['mrk']['pos'][0][0][0]
from scipy.signal import butter,lfilter
import numpy as np
sample=np.zeros((200,59,500))
for i in range(200):
    for j in range(59):
        sample[i][j][:]=ss[j][pos[i]:pos[i]+500]
b,a=butter(5,[8/50,30/50],btype='bandpass')
samplefilted=np.ones((200,59,500))
for i in range(200):
    for j in range(59):
        samplefilted[i][j][:]=lfilter(b,a,sample[i][j][:])
#--------------10-fold----------------
class1=np.zeros((100,59,500))
class2=np.zeros((100,59,500))
count1=0
count2=0
for i in range(200):
    if y[i]>0:
        class1[count1]=samplefilted[i]
        count1+=1
    else:
        class2[count2]=samplefilted[i]
        count2+=1
test_samplefilted_class1=np.zeros((10,59,500))
train_samplefilted_class1=np.zeros((90,59,500))
test_samplefilted_class2=np.zeros((10,59,500))
train_samplefilted_class2=np.zeros((90,59,500))
a=range(100)
b1=np.reshape(a,(10,10))
acc=np.zeros((10))
for k in range(10):
    b2=np.setdiff1d(a,b1[k])
    test_samplefilted_class1=class1[b1[k]]
    train_samplefilted_class1=class1[b2]
    test_samplefilted_class2=class2[b1[k]]
    train_samplefilted_class2=class2[b2]
    #-----------寻找最佳起始点-----------
    import startpoint_find
    shili=startpoint_find.main(train_samplefilted_class1,train_samplefilted_class2)
    startpoint=shili.csp()
    #startpoint=200
    #--------寻找参考信号---------------
    import find_refrence
    reference=find_refrence.main(train_samplefilted_class1,train_samplefilted_class2,startpoint)
    c1_r1,c1_r2,c2_r1,c2_r2=reference.csp()
    #---------训练模型------------------
    import train_model
    model=train_model.main(train_samplefilted_class1,train_samplefilted_class2,c1_r1,c1_r2,c2_r1,c2_r2,startpoint)
    F1,F2,svm=model.csp()
    a1=np.zeros((10,200))
    a2=np.zeros((10,200))
    max1=np.zeros((10))
    max2=np.zeros((10))
    test11=np.zeros((10,59,200))
    test12=np.zeros((10,59,200))
    test21=np.zeros((10,59,200))
    test22=np.zeros((10,59,200))
    for i in range(10):
        for j in range(100):
            a1[i,j]=np.cov(c1_r1,test_samplefilted_class1[i,26,startpoint+j-50:startpoint+j-50+200])[0,1]+np.cov(c1_r2,test_samplefilted_class1[i,30,startpoint+j-50:startpoint+j-50+200])[0,1]
            a2[i,j]=np.cov(c2_r1,test_samplefilted_class2[i,26,startpoint+j-50:startpoint+j-50+200])[0,1]+np.cov(c2_r2,test_samplefilted_class2[i,30,startpoint+j-50:startpoint+j-50+200])[0,1]
        max1[i]=np.argmax(a1[i,:])
        max2[i]=np.argmax(a1[i,:])
        test11[i]=test_samplefilted_class1[i,:,np.int(startpoint+max1[i]-50):np.int(startpoint+max1[i]-50+200)]
        test12[i]=test_samplefilted_class1[i,:,np.int(startpoint+max2[i]-50):np.int(startpoint+max2[i]-50+200)]
        test21[i]=test_samplefilted_class2[i,:,np.int(startpoint+max1[i]-50):np.int(startpoint+max1[i]-50+200)]
        test22[i]=test_samplefilted_class2[i,:,np.int(startpoint+max2[i]-50):np.int(startpoint+max2[i]-50+200)]
    #----------投影矩阵提取特征---------------------
    feature1=np.zeros((10,8))
    feature2=np.zeros((10,8))
    for i in range(10):
        dataf11=np.dot(test11[i].T,F1.T)
        dataf12=np.dot(test12[i].T,F2.T)
        dataf21=np.dot(test21[i].T,F1.T)
        dataf22=np.dot(test22[i].T,F2.T)
        for j in range(4):
            feature1[i,j]=np.log(np.var(dataf11[:,j]))
            feature1[i,j+4]=np.log(np.var(dataf12[:,j]))
            feature2[i,j]=np.log(np.var(dataf21[:,j]))
            feature2[i,j+4]=np.log(np.var(dataf22[:,j]))
    feature=np.vstack((feature1,feature2))
    #---------------分类---------------------
    predict_label=svm.predict(feature)
    testlabel=np.vstack((np.ones((10,1)),np.ones((10,1))*-1))
    acc[k]=sum(predict_label==np.squeeze(testlabel))/20
print(np.mean(acc))