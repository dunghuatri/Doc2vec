import sys

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

#glob
import glob

import random

#pandas
import pandas as pd

import timeit
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
            for file_ in allFiles:
                df = pd.read_csv(file_)
                content = df['content']
                title = df['title']
                print('number documents: ', len(content))
                for row in content:
                    self.sentences.append(TaggedDocument(utils.to_unicode(row).split(), [prefix + '_%s' % item_no]))
                    item_no = item_no + 1
        return self.sentences, title

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled
#---------------------------------------------------------------------------------------------------------------------#
def Save_news_title(data_path,file_name):
    # ===============================#
    # Xét đường dẫn tới dataset
    sources = {data_path: 'SAMPLE', }
    print('Load data ...')
    start = timeit.default_timer()
    sentences = LabeledContent(sources)
    stop = timeit.default_timer()
    print('Done! time: ', stop - start, ' (s)')
    # ===============================#

    print('Tokenize ...')
    start = timeit.default_timer()
    token,title = sentences.to_array()
    stop = timeit.default_timer()
    print('Done! time: ', stop - start, ' (s)')

    print('Saving title ...')
    start = timeit.default_timer()
    title.to_csv(file_name)
    stop = timeit.default_timer()
    print('Done! time: ', stop - start, ' (s)')

if __name__== "__main__":
    Save_news_title(sys.argv[1],sys.argv[2])