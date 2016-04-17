from classifyUtils import *
from nltk.classify import NaiveBayesClassifier
import pandas as pd
import numpy as np
import math
import sys


# Global constants
verbose = True
train_sample_rate = 0.9
input_col = 'rev_text_lemm'
# DEBUG coded_lines temp var until all lines have been coded
# coded_lines = sys.argv[2] if len(sys.argv) > 2 else 4998
feature = eval(sys.argv[1]) if len(sys.argv) > 1 else unigram_boolean_features

# df = pd.read_excel('../proc/sentences_lemm_lab.xlsx')[1:coded_lines+1]
df = pd.read_excel('../proc/sentences_lemm_lab.xlsx')[1:]
d = {'dimension': [], 'neg_train':[], 'pos_train':[], 'mif': [], 'pos_accuracy': [], 'pos_precision': [], 'pos_recall': [], 'pos_fmeasure': [], 'neg_accuracy': [], 'neg_precision': [], 'neg_recall': [], 'neg_fmeasure': [], 'ave_accuracy': [], 'ave_precision': [], 'ave_recall': [], 'ave_fmeasure': []}

# Randomise the order of the lines
# df = df.iloc[np.random.permutation(len(df))]

class_labels = (0, -1, 1)   # useful labels being (-1, 1)
dimensions = df.columns.values[5:]  # DEBUG array is [5:6] else should read [5:]

for dimension in dimensions:
    # split the data into class labels (e.g., pos and neg, or more), approx 10% for test, 90% for train
    data = dict(pos = df[df[dimension].isin(class_labels[1:])], neg = df[~df[dimension].isin(class_labels[1:])])

    # get number of samples for the training sets
    (neg_train, pos_train) = (
        sample_dataset(
            data['neg'][dimension], train_sample_rate), 
        sample_dataset(
            data['pos'][dimension], train_sample_rate))

    # Extract the features
    (neg_training, neg_test, pos_training, pos_test) = (
        extract_features(data['neg'][input_col][:neg_train], 'neg', feature_extractor=feature),
        extract_features(data['neg'][input_col][neg_train:], 'neg', feature_extractor=feature),
        extract_features(data['pos'][input_col][:pos_train], 'pos', feature_extractor=feature),
        extract_features(data['pos'][input_col][pos_train:], 'pos', feature_extractor=feature)
        )

    train_set = pos_training + neg_training
    test_set = pos_test + neg_test

    # train the NaiveBayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    # Run evaluations
    if verbose:
        print_basic_info(dimension, feature.__name__, (len(neg_test), len(pos_test), len(neg_training), len(pos_training)))
    mif = show_most_informative_features(classifier, 15)
    for line in mif:
        print line
    results = do_evaluation(test_set, classifier, verbose=verbose)

    # Store into dict d for df
    d['dimension'].append(dimension)
    d['neg_train'].append(neg_train)
    d['pos_train'].append(pos_train)
    d['mif'].append(mif[1:])
    d['pos_accuracy'].append(results['pos']['accuracy'])
    d['pos_precision'].append(results['pos']['precision'])
    d['pos_recall'].append(results['pos']['recall'])
    d['pos_fmeasure'].append(results['pos']['fmeasure'])

    d['neg_accuracy'].append(results['neg']['accuracy'])
    d['neg_precision'].append(results['neg']['precision'])
    d['neg_recall'].append(results['neg']['recall'])
    d['neg_fmeasure'].append(results['neg']['fmeasure'])

    d['ave_accuracy'].append(results['ave']['accuracy'])
    d['ave_precision'].append(results['ave']['precision'])
    d['ave_recall'].append(results['ave']['recall'])
    d['ave_fmeasure'].append(results['ave']['fmeasure'])

out_df = pd.DataFrame(d, columns=['dimension', 'neg_train', 'pos_train', 'mif', 'pos_accuracy', 'pos_precision', 'pos_recall', 'pos_fmeasure', 'neg_accuracy', 'neg_precision', 'neg_recall', 'neg_fmeasure', 'ave_accuracy', 'ave_precision', 'ave_recall', 'ave_fmeasure'])

out_df.to_excel('../proc/perf_nb_' + feature.__name__ + '.xlsx', index=False)