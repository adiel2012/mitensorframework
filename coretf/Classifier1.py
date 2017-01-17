import numpy as np
from IClassifier import *
from Experiment import *

class Classifier1(IClassifier):
#    def __init__(self):

    @staticmethod
    def buildClassifier(data):
        pass
    
    @staticmethod
    def probabilities(row):
        return np.array([0, 1, 0]); 
    
    


cl =  Classifier1();
X=np.zeros((10,4));
Y=np.array([1,1,0,1,1,1,1,1,2,1]);

experiment = Experiment();
acc = experiment.accuracy(cl, X, Y, 3);
print(acc);

acc = experiment.confussion_matrix(cl, X, Y, 3);
print(acc);

