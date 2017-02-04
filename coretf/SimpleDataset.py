class SimpleDataset:
    def __init__(self,x,y):
        self.x = x;
        self.y = y;

    def getX(self):
        return self.x;
    def getY(self):
        return self.y;

    def getNumclasses(self):
        return len(self.y[0]);

    def getNumPatterns(self):
        return len(self.x);

    def getNumAttrs(self):
        return len(self.x[0]);

    def at(self, index):
        return self.x[index];

inst = SimpleDataset([[1, 2, 5], [3, 4, 345]],[[1, 0], [0, 1]]);
print(inst.getX())
print(inst.getNumclasses())

print(inst.at(1))
print(inst.getNumPatterns())