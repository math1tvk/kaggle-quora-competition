

"""module docstring"""


# imports

import sys

import gensim
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import Imputer

# import os
# import csv
import re
from nltk.corpus import stopwords
import numpy as np

# constants
# exception classes
# interface functions
# classes
# internal functions & classes



def question2words(question, stops):
    """
    :param question: single question string
    :return:
    This function converts a raw question to a string of words
    """
    # remove non-letters => C vs C++
    letters_only = re.sub("[^a-zA-Z]", " ", question)

    # convert to lower case, split into separate words
    words = letters_only.lower().split(" ")

    # remove stop words
    meaningful_words = [w for w in words if (not (w in stops or len(w)<2))]

    # return an array of meaningful words
    return (meaningful_words)

def qwords2vector(words, model, index2word_set, num_features):
    """
    Function to average all of the word vectors in a given question

    :param words:
    :param model:
    :param index2word_set:
    :param num_features:
    :return:
    """

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0

    # Loop over each word in the question and, if it is in the model's
    # vocaublary, add its feature vector to the total

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords == 0:
        return featureVec

    featureVec = np.divide(featureVec,nwords)
    return featureVec

def dot(x,y):
    return [abs(x1-y1) for x1,y1 in zip(x,y)]

# Main
def main():

    ## embedding model
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)



    ## read data
    train_data = pd.read_csv("../Data/train.csv")
    test_data = pd.read_csv("../Data/test.csv")

    ## separate input and result

    # columns: ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate']
    train_id = train_data['id']
    train_qid1 = train_data['qid1']
    train_qid2 = train_data['qid2']
    train_question1 = train_data['question1']
    train_question2 = train_data['question2']
    train_is_duplicate = train_data['is_duplicate']

    # columns: ['id', 'qid1', 'qid2', 'question1', 'question2']
    test_id = test_data['test_id']
    test_question1 = test_data['question1']
    test_question2 = test_data['question2']

    ## clean the input
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))

    clean_train_question1 = [question2words(str(x),stops) for x in train_question1]
    clean_train_question2 = [question2words(str(x),stops) for x in train_question2]

    clean_test_question1 = [question2words(str(x),stops) for x in test_question1]
    clean_test_question2 = [question2words(str(x),stops) for x in test_question2]

    #clean_data_question2 = map(lambda x:question2words(x,stops),data_question2)

    # removing repeated words?


    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    num_features = 300


    ## convert to vector
    vectors_train_question1 = [qwords2vector(x, model, index2word_set, num_features) for x in clean_train_question1]
    vectors_train_question2 = [qwords2vector(x, model, index2word_set, num_features) for x in clean_train_question2]

    vectors_test_question1 = [qwords2vector(x, model, index2word_set, num_features) for x in clean_test_question1]
    vectors_test_question2 = [qwords2vector(x, model, index2word_set, num_features) for x in clean_test_question2]


    ## compute the distance of question 1 and question 2
    train_distance_q1_q2 = [dot(x,y) for x,y in zip(vectors_train_question1,vectors_train_question2)]
    test_distance_q1_q2 = [dot(x,y) for x,y in zip(vectors_test_question1,vectors_test_question2)]

    # just for now, because of NAN error, should be resolved in better way
    train_features = train_distance_q1_q2 # Imputer().fit_transform(train_distance_q1_q2)
    train_result = train_is_duplicate

    test_features = test_distance_q1_q2 # Imputer().fit_transform(test_distance_q1_q2)



    # as we only have the label of train data, we test it by cross validation in train data
    train_train_data, train_test_data, train_train_result, train_test_result = cross_validation.train_test_split(train_features,train_result, test_size = 0.2, random_state = 42322)

    # Fit a random forest to the training data, using 1000 trees
    forest = RandomForestClassifier(n_estimators = 100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_train_data, train_train_result)


    score_tr_tr = forest.score(train_train_data, train_train_result) #0.98792018105815138
    score_tr_tst = forest.score(train_test_data, train_test_result) #0.62979544386455266 => This is so-called overfitting :D

    result_prob_tr_tr = forest.predict_proba(train_train_data)
    result_prob_tr_tst = forest.predict_proba(train_test_data)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_test_result, result_prob_tr_tst[:, 1])
    roc_auc = auc(false_positive_rate, true_positive_rate) #0.66166516195753156

    print("train score = %s, test score = %s, roc_auc = %s",str(score_tr_tr),str(score_tr_tst),str(roc_auc))

    ## here is the main testing for the main result ##
    # Fit a random forest to the training data, using 1000 trees
    # forest = RandomForestClassifier(n_estimators = 100)

    # print("Fitting a random forest to labeled training data...")
    # forest = forest.fit(train_features, train_result)

    # result_tst = forest.predict(test_features)
    # result_prob_tst = forest.predict_proba(test_features)


if __name__ == '__main__':
    status = main()
    sys.exit(status)