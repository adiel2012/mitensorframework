import numpy as np
from sklearn.model_selection import StratifiedKFold

class Experiment:
#    def __init__(self):


    @staticmethod
    def confussion_matrix(aclassifier,X,Y,num_classes):
        ashape = X.shape;
        num_patterns = len(Y);
        mat = np.zeros((num_classes,num_classes));
        
        for i in range(num_patterns):
            probs = aclassifier.probabilities(X[i]);
            class_predicted = np.argmax(probs);
            cl_index = np.argmax(Y[i]);
            mat[cl_index,class_predicted] = mat[cl_index,class_predicted] + 1;
        
        #print(mat)
        return mat;
    
    @staticmethod
    def accuracy(aclassifier,X,Y,num_classes):
        mat = Experiment.confussion_matrix(aclassifier,X,Y,num_classes);
        sum = 0;
        for i in range(num_classes):
            sum = sum + mat[i,i];
        num_patterns = len(Y);
        return sum/num_patterns;
        
    @staticmethod
    def toIndex(x):
        res = [];
        for row in x:
            res.append(np.argmax(row));
        return res;
    
    @staticmethod   
    def Kfold(x,y,K):
        skf = StratifiedKFold(n_splits=K,shuffle=True, random_state=0)
        Y_index = Experiment.toIndex(y);
        skf.get_n_splits(x, Y_index)
        res=[];
        for train_index, test_index in skf.split(x, Y_index):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            res.append([X_train, X_test, y_train, y_test]);
        return res;
        
