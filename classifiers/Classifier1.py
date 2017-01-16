import numpy as np

class IClassifier(core.IClassifier):
#    def __init__(self):

    def buildClassifier(self,data):
        pass

    def probabilities(self,row):
        return np.array([0, 1, 0]); 