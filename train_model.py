class main(object):
    def __init__(self,train_samplefilted_class1,train_samplefilted_class2,c1_r1,c1_r2,c2_r1,c2_r2,startpoint):
        self.class1=train_samplefilted_class1
        self.class2=train_samplefilted_class2
        self.c1_r3=c1_r1
        self.c1_r4=c1_r2
        self.c2_r3=c2_r1
        self.c2_r4=c2_r2
        self.startpoint=startpoint
    def csp(self):
        import numpy as np
        label=np.vstack((np.ones((90,1)),np.ones((90,1))*-1))
        class1_adjust_c1=np.zeros((90,59,200))
        class2_adjust_c1=np.zeros((90,59,200))
        class1_adjust_c2=np.zeros((90,59,200))
        class2_adjust_c2=np.zeros((90,59,200))
        #-------------------------c1-----------------
        a1=np.zeros((90,200))
        a2=np.zeros((90,200))
        max1=np.zeros((90))
        max2=np.zeros((90))
        for i in range(90):
            for j in range(100):
                a1[i,j]=np.cov(self.c1_r3,self.class1[i,26,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]+np.cov(self.c1_r4,self.class1[i,30,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]
                a2[i,j]=np.cov(self.c1_r3,self.class2[i,26,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]+np.cov(self.c1_r4,self.class2[i,30,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]
            max1[i]=np.argmax(a1[i])
            max2[i]=np.argmax(a2[i])
        for i in range(90):
            class1_adjust_c1[i]=self.class1[i,:,np.int(self.startpoint+max1[i]-50):np.int(self.startpoint+max1[i]-50+200)]
            class2_adjust_c1[i]=self.class2[i,:,np.int(self.startpoint+max2[i]-50):np.int(self.startpoint+max2[i]-50+200)]
        F1,feature1=self.csp_sub(class1_adjust_c1,class2_adjust_c1)
        #-----------------------------c2---------------------------------------------
        a1=np.zeros((90,200))
        a2=np.zeros((90,200))
        max1=np.zeros((90))
        max2=np.zeros((90))
        for i in range(90):
            for j in range(100):
                a1[i,j]=np.cov(self.c2_r3,self.class1[i,26,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]+np.cov(self.c2_r4,self.class1[i,30,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]
                a2[i,j]=np.cov(self.c2_r3,self.class2[i,26,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]+np.cov(self.c2_r4,self.class2[i,30,np.int(self.startpoint+j-50):np.int(self.startpoint+j-50+200)])[0,1]
            max1[i]=np.argmax(a1[i])
            max2[i]=np.argmax(a2[i])
        for i in range(90):
            class1_adjust_c2[i]=self.class1[i,:,np.int(self.startpoint+max1[i]-50):np.int(self.startpoint+max1[i]-50+200)]
            class2_adjust_c2[i]=self.class2[i,:,np.int(self.startpoint+max2[i]-50):np.int(self.startpoint+max2[i]-50+200)]
        F2,feature2=self.csp_sub(class1_adjust_c2,class2_adjust_c2)
        #----------------------------svm-----------------------------
        from sklearn.svm import SVC
        svm=SVC(kernel='linear',C=1.0,random_state=0)
        svm.fit(np.hstack((feature1,feature2)),np.vstack((np.ones((90,1)),np.ones((90,1))*-1)))
        return F1,F2,svm
    def csp_sub(self,class1,class2):
        import numpy as np
        R1 = np.zeros((59, 59))
        R2 = np.zeros((59, 59))
        for i in range(90):
            R1 = R1 + np.cov(class1[i])
            R2 = R2 + np.cov(class2[i])
        R1 = R1 / 90
        R2 = R2 / 90
        R3 = R1 + R2
        Sigma, U0 = np.linalg.eig(R3)
        P = np.dot(np.diag(Sigma ** (-0.5)), U0.T)
        YL = np.dot(np.dot(P, R1), P.T)
        SigmaL, UL = np.linalg.eig(YL)
        I = np.argsort(SigmaL)
        F1 = np.dot(UL.T, P)[[0, 1, 57, 58], :]
        feature1=np.zeros((90,4))
        feature2=np.zeros((90,4))
        for i in range(90):
            dataf11=np.dot(class1[i].T,F1.T)
            dataf12=np.dot(class2[i].T,F1.T)
            for j in range(4):
                feature1[i,j]=np.log(np.var(dataf11[:,j]))
                feature2[i,j]=np.log(np.var(dataf12[:,j]))
        feature=np.vstack((feature1,feature2))
        return F1,feature