class main(object):
    def __init__(self,class1,class2):
        self.class1=class1
        self.class2=class2
    def csp(self):
        import numpy as np
        pre_startpoint=np.array([50,100,150,200,250])
        ACC=np.zeros((5))
        count=0
        for startpoint in pre_startpoint:
            acc=np.zeros((10))
            sample_class1=self.class1[:,:,startpoint:startpoint+200]
            sample_class2=self.class2[:,:,startpoint:startpoint+200]
            a=range(90)
            b1=np.reshape(a,(10,9))
            for k in range(10):
                b2=np.setdiff1d(a,b1[k])
                class1_train=sample_class1[b2]
                class2_train=sample_class2[b2]
                class1_test=sample_class1[b1[k]]
                class2_test=sample_class2[b1[k]]
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
                feature1 = np.zeros((81, 4))
                feature2 = np.zeros((81, 4))
                tes1 = np.zeros((9, 4))
                tes2 = np.zeros((9, 4))
                for i in range(81):
                    dataf11 = np.dot(class1_train[i].T, F1.T)
                    dataf12 = np.dot(class2_train[i].T, F1.T)
                    for j in range(4):
                        feature1[i, j] = np.log(np.var(dataf11[:, j]))
                        feature2[i, j] = np.log(np.var(dataf12[:, j]))
                for i in range(9):
                    dataf11 = np.dot(class1_test[i].T, F1.T)
                    dataf12 = np.dot(class2_test[i].T, F1.T)
                    for j in range(4):
                        tes1[i, j] = np.log(np.var(dataf11[:, j]))
                        tes2[i, j] = np.log(np.var(dataf12[:, j]))
                traindata = np.vstack((feature1, feature2))
                testdata = np.vstack((tes1, tes2))
                from sklearn.svm import SVC
                svm = SVC(kernel='linear', C=1.0, random_state=0)
                svm.fit(traindata, np.vstack((np.ones((81, 1)), np.ones((81, 1)) * -1)))
                predict_label = svm.predict(testdata)
                acc[k] = np.sum(predict_label == np.squeeze(np.vstack((np.ones((9, 1)), np.ones((9, 1)) * -1)))) / 18
            ACC[count]=np.mean(acc)
            count+=1
        return (np.argmax(ACC)+1)*50
