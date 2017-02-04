from coretf.SimpleDataset import *
from coretf.ISimpleDatasetProvider import *
import re
import numpy as np
from tensorflow.contrib import learn

class DatasetProviderText1(ISimpleDatasetProvider):
    
    
    def clean_str(self,string):

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    
    def load_data_and_labels(self,positive_data_file, negative_data_file):

    # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
    # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
    # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        return [x_text, y]


    def getISimpleDataset(self):
        
        x_text, y = self.load_data_and_labels("../data/rt-polaritydata/rt-polarity.pos","../data/rt-polaritydata/rt-polarity.neg")
        max_document_length = max([len(x.split(" ")) for x in x_text]);
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = list(vocab_processor.fit_transform(x_text))
        x = np.array(x)
        #print(type(x));
        
        return SimpleDataset(x,y);
        #print(x)
        #return SimpleDataset([[1, 2, 5], [3, 4, 345]],[[1, 0], [0, 1]]);





#inst1 = DatasetProviderText1();
#inst = inst1.getISimpleDataset();
#print(inst.getX())
#print(inst.getNumclasses())
#print(inst.at(1))
#print(inst.getNumPatterns())