class main(object):
    def __init__(self,class1,class2,startpoint):
        self.class1=class1
        self.class2=class2
        self.startpoint=startpoint
    def csp(self):
        import numpy as np
        c1_r3=np.zeros((200))
        c1_r4=np.zeros((200))
        c2_r3=np.zeros((200))
        c2_r4=np.zeros((200))
        ACC=0
        pos1=np.ones((90))*self.startpoint
        pos2=np.ones((90))*self.startpoint
        for i in range(90):
            c1_r3=c1_r3+self.class1[i,26,self.startpoint:self.startpoint+200]
            c1_r4=c1_r4+self.class1[i,30,self.startpoint:self.startpoint+200]
            c2_r3=c2_r3+self.class2[i,26,self.startpoint:self.startpoint+200]
            c2_r4=c2_r4+self.class2[i,30,self.startpoint:self.startpoint+200]
        while 1:
            a1=np.zeros((90,20))
            a2=np.zeros((90,20))
            max1=np.zeros((90))
            max2=np.zeros((90))
            for i in range(90):
                for j in range(20):
                    a1[i,j]=np.cov(c1_r3,self.class1[i,26,np.int(pos1[i]+j-10):np.int(pos1[i]+j-10+200)])[0,1]+np.cov(c1_r4,self.class1[i,30,np.int(pos1[i]+j-10):np.int(pos1[i]+j-10+200)])[0,1]
                    a2[i,j]=np.cov(c2_r3,self.class2[i,26,np.int(pos2[i]+j-10):np.int(pos2[i]+j-10+200)])[0,1]+np.cov(c2_r4,self.class2[i,30,np.int(pos2[i]+j-10):np.int(pos2[i]+j-10+200)])[0,1]
                max1[i]=np.argmax(a1[i])
                pos1[i]=pos1[i]+max1[i]-10
                max2[i]=np.argmax(a2[i])
                pos2[i]=pos2[i]+max2[i]-10
            adjust_class1=np.zeros((90,59,200))
            adjust_class2=np.zeros((90,59,200))
            for i in range(90):
                adjust_class1[i]=self.class1[i,:,np.int(pos1[i]):np.int(pos1[i]+200)]
                adjust_class2[i]=self.class2[i,:,np.int(pos2[i]):np.int(pos2[i]+200)]
            a=range(90)
            b1=np.reshape(a,(10,9))
            acc=np.zeros((10))
            for k in range(10):
                b2=np.setdiff1d(a,b1[k])
                class1_train=adjust_class1[b2]
                class2_train=adjust_class2[b2]
                class1_test=adjust_class1[b1[k]]
                class2_test=adjust_class2[b1[k]]
                R1=np.zeros((59,59))
                R2=np.zeros((59,59))
                for i in range(81):
                    R1=R1+np.cov(class1_train[i])
                    R2=R2+np.cov(class2_train[i])
                R1=R1/81
                R2=R2/81
                R3=R1+R2
                Sigma, U0 = np.linalg.eig(R3)
                P = np.dot(np.diag(Sigma ** (-0.5)), U0.T)
                YL = np.dot(np.dot(P, R1), P.T)
                SigmaL, UL = np.linalg.eig(YL)
                I = np.argsort(SigmaL)
                F1 = np.dot(UL.T, P)[[0, 1, 57, 58], :]
                feature1=np.zeros((81,4))
                feature2=np.zeros((81,4))
                tes1=np.zeros((9,4))
                tes2=np.zeros((9,4))
                for i in range(81):
                    dataf11=np.dot(class1_train[i].T,F1.T)
                    dataf12=np.dot(class2_train[i].T,F1.T)
                    for j in range(4):
                        feature1[i,j]=np.log(np.var(dataf11[:,j]))
                        feature2[i,j]=np.log(np.var(dataf12[:,j]))
                for i in range(9):
                    dataf11=np.dot(class1_test[i].T,F1.T)
                    dataf12=np.dot(class2_test[i].T,F1.T)
                    for j in range(4):
                        tes1[i,j]=np.log(np.var(dataf11[:,j]))
                        tes2[i,j]=np.log(np.var(dataf12[:,j]))
                traindata=np.vstack((feature1,feature2))
                testdata=np.vstack((tes1,tes2))
                from sklearn.svm import SVC
                svm=SVC(kernel='linear',C=1.0,random_state=0)
                svm.fit(traindata,np.vstack((np.ones((81,1)),np.ones((81,1))*-1)))
                predict_label=svm.predict(testdata)
                acc[k]=np.sum(predict_label==np.squeeze(np.vstack((np.ones((9,1)),np.ones((9,1))*-1))))/18
            if np.mean(acc)>=ACC:
                ACC=np.mean(acc)
                c1_r3=np.zeros((200))
                c1_r4=np.zeros((200))
                c2_r3=np.zeros((200))
                c2_r4=np.zeros((200))
                for i in range(90):
                    c1_r3=c1_r3+adjust_class1[i,26]
                    c1_r4=c1_r4+adjust_class1[i,30]
                    c2_r3=c2_r3+adjust_class2[i,26]
                    c2_r4=c2_r4+adjust_class2[i,30]
            else:
                break
        return c1_r3,c1_r4,c2_r3,c2_r4
