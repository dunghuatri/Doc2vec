import sys

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

#glob
import glob

# numpy
import numpy as np

#pandas
import pandas as pd

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# random, itertools, matplotlib
import random

import multiprocessing

import timeit
import itertools

#---------------------------------------------------------------------------------------------------------------------#
# Class đọc từng dòng 'content' trong file csv
class LabeledContent(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def to_array(self):
        self.sentences = []
        item_no = 0
        for source, prefix in self.sources.items():

            allFiles = glob.glob(source + "/*.csv")
            series_ = pd.DataFrame()
            list_ = []
            for file_ in allFiles:
                df = pd.read_csv(file_)
                content = df['content']
                list_.append(content)
            series_ = pd.concat(list_)
            series_ = series_.drop_duplicates()
            series_ = series_.dropna()
            for row in series_:
                self.sentences.append(TaggedDocument(utils.to_unicode(row).split(), [prefix + '_%s' % item_no]))
                item_no = item_no + 1
        return self.sentences

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled
#---------------------------------------------------------------------------------------------------------------------#
def Train_doc2vec_model(path,model_name):
    # ===============================#
    # Xét đường dẫn tới dataset
    sources = {path: 'SAMPLE', }
    sentences = LabeledContent(sources)
    # ===============================#
    # Xét tham số cho model, build vocabulary
    cores = multiprocessing.cpu_count()
    print('Num of cores is %s' % cores)
    model = Doc2Vec(min_count=5, window=10, vector_size=400, sample=1e-4, negative=5, workers=10, dm=0)
    print('Build vocabulary ...')
    model.build_vocab(sentences.to_array())
    print('Done!')
    # ===============================#
    # Train model
    print('Train model ...')
    start = timeit.default_timer()
    model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=1)
    stop = timeit.default_timer()
    print('Done! time: ', stop - start, ' (s)')
    # ===============================#
    # Save model
    print('Save model ...')

    model.save(model_name)
    print('Done!')
if __name__== "__main__":
    Train_doc2vec_model(sys.argv[1],sys.argv[2])