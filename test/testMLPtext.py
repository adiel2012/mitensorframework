from classifier.ClassifierMLP import *
from dataproviders.DatasetProviderText1 import * 
from coretf.Experiment import *
'''
Created on Feb 4, 2017

@author: adiel
'''



cl =  ClassifierMLP();


inst1 = DatasetProviderText1();
inst = inst1.getISimpleDataset();
print(inst.getX())
print(inst.getNumclasses())
print(inst.at(1))
print(inst.getNumPatterns())

X= inst.getX() ; #np.zeros((10,4));
Y= inst.getY() ; # np.array([1,1,0,1,1,1,1,1,2,1]);

cl.buildClassifier(X, Y);

index = 0;
prob = cl.probabilities(X[index]);
print(prob);
print(Y[index]);
#print(np.nargmax(Y, axis=0))
print("----------------------------------------------------")

cm = Experiment.confussion_matrix(cl, X, Y, inst.getNumclasses());
print(cm);
acc = Experiment.accuracy(cl, X, Y, inst.getNumclasses());
print(acc);

print("----------------------------------------------------")
K=4;
accSum = 0;
partitions = Experiment.Kfold(X, Y, K)
for partition in partitions:
    X_train = partition[0];
    X_test = partition[1];
    y_train = partition[2];
    y_test = partition[3];
    cl.buildClassifier(X_train, y_train);
    num_instances = len(y_train[0]);
    cm = Experiment.confussion_matrix(cl, X_test, y_test, num_instances);
    accSum = accSum + Experiment.accuracy(cl, X_test, y_test, num_instances);
    
    
    
accSum = accSum/K;
print("precision")
print(accSum);    
    #print(cm);
    
#experiment = Experiment();
#acc = experiment.accuracy(cl, X, Y, 3);
#print(acc);

#acc = experiment.confussion_matrix(cl, X, Y, 3);
#print(acc);

