import numpy as np
class Experiment:
#    def __init__(self):


    def confussion_matrix(self,aclassifier,X,Y,num_classes):
        ashape = X.shape;
        num_patterns = len(Y);
        mat = np.zeros((num_classes,num_classes));
        
        for i in range(num_patterns):
            probs = aclassifier.probabilities(X[i]);
            class_predicted = np.argmax(probs);
            mat[Y[i],class_predicted] = mat[Y[i],class_predicted] + 1;
        
        #print(mat)
        return mat;
    
    
    def accuracy(self,aclassifier,X,Y,num_classes):
        mat = self.confussion_matrix(aclassifier,X,Y,num_classes);
        sum = 0;
        for i in range(num_classes):
            sum = sum + mat[i,i];
        num_patterns = len(Y);
        return sum/num_patterns;
        
        
