from coretf.SimpleDataset import *
from coretf.ISimpleDatasetProvider import *

class SimpleDatasetProvider1(ISimpleDatasetProvider):

    def getISimpleDataset(self):
        def __init__(self):
            pass;

        return SimpleDataset([[1, 2, 5], [3, 4, 345]],[[1, 0], [0, 1]]);





inst1 = SimpleDatasetProvider1();
inst = inst1.getISimpleDataset();
print(inst.getX())
print(inst.getNumclasses())
print(inst.at(1))
print(inst.getNumPatterns())