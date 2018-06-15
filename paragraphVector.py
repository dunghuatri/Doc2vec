#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import sys
import csv
import pandas as pd


def main():
    train_model()
    model = Doc2Vec.load("d2v.model")
    # to find the vector of a document which is not in training data
    test_data = word_tokenize("I love chatbots".lower())
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)

    # to find most similar doc using tags
    # similar_doc = model.docvecs.most_similar('1')
    # print(similar_doc)

    # to find vector of doc in training data using tags or in other words
    # , printing the vector of document at index 1 in training data
    # print(model.docvecs['1'])


def train_model():
    input_dir = "data/2015-news-7.1-8.31.csv"
    small_dir = "data/small.csv"
    # data = ["I love machine learning. Its awesome.",
    #         "I abc coding in python",
    #         "I hate building chatbots",
    #         "they chat amagingly well"]

    # set max field size
    csv.field_size_limit(sys.maxsize)
    # load file from csv
    documents = []
    tags = []
    # csv format id,title,content,source,create_time,get_time
    # with open(input_dir) as csvDataFile:
    #     csvReader = csv.reader(csvDataFile)
    #     next(csvReader, None)  # skip the headers
    #     for row in csvReader:
    #         documents.append(row[2])
    #         tags.append(row[0])

    # read csv using pandas

    # input_dir = "data/2015-news-7.1-8.31.csv"
    # for df in pd.read_csv(input_dir, sep=',', header=0, chunksize=5, encoding="utf-8"):
    #     tags.append(df["id"].astype(str).values[0])
    #     documents.append(df["content"].astype(str).values[0])

    # test another pandas

    df2 = pd.read_csv(input_dir, sep=',', header=0, encoding="utf-8")
    print("Finish read csv")
    tags = df2["id"].astype(str).values
    documents = df2["content"].astype(str).values

    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[tags[i]]) for i, _d in enumerate(documents)]
    print("Finish load tagged data")
    max_epochs = 10
    vec_size = 300
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Model Saved")


if "__name__": main()
