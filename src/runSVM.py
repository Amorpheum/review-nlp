from classifyUtils import *
import nltk.classify
from sklearn import svm, metrics
import pandas as pd
import numpy as np
import math
import sys
import pickle

def train_svm(kernel, C, class_weight, kfolds, d):
    # data = {'pos': , 'neg': }
    neg_training = extract_features(data['neg'][input_col], 'neg', feature_extractor=feature)
    pos_training = extract_features(data['pos'][input_col], 'pos', feature_extractor=feature)
    train_set = pos_training + neg_training
    
    # train the SVM
    cl = svm.SVC(
        kernel=kernel,
        class_weight=class_weight,
        C = C
    )
    clf = nltk.classify.SklearnClassifier(cl)
    clf.train(train_set)

    # save the pickle
    f = open('../pickles/' + dimension + '_clf.pickle', 'wb')
    pickle.dump(clf, f)
    f.close()

    # Working test load and run the pickle
    # f = open('../pickles/' + dimension + '_clf.pickle', 'rb')
    # clf = pickle.load(f)
    # f.close()

    # efs = extract_features(df[input_col], feature_extractor=feature)

    # correct = 0    
    # for ef, actual in zip(efs, df[dimension]):
    #     res = clf.classify(ef)

    #     if res == 'pos':
    #         pred = 1.0
    #     else:
    #         pred = 0.0

    #     if pred == actual == 1.0:
    #         correct += 1
    #         print ef, actual
    # print 'Correct:', correct, 'Total:', len(efs)

def run_svm_with_params(kernel, C, class_weight, kfolds, d):
    try:
        # Split data for cross validation
        folded = cross_validation_folds(data, dimension, kfolds)
        cvresults = []

        # train the SVM
        cl = svm.SVC(
            kernel=kernel,
            class_weight=class_weight,
            C = C
            )
        clf = nltk.classify.SklearnClassifier(cl)

        for f in range(kfolds):

            neg_training = extract_features(df.ix[folded['neg'][f]['train'].index][input_col], 'neg', feature_extractor=feature)
            pos_training = extract_features(df.ix[folded['pos'][f]['train'].index][input_col], 'pos', feature_extractor=feature)
            neg_test = extract_features(df.ix[folded['neg'][f]['test'].index][input_col], 'neg', feature_extractor=feature)
            pos_test = extract_features(df.ix[folded['pos'][f]['test'].index][input_col], 'pos', feature_extractor=feature)

            train_set = pos_training + neg_training
            test_set = pos_test + neg_test

            # Train the classifier
            clf.train(train_set)

            # Run evaluations
            if verbose:
                print_basic_info(dimension, feature.__name__, (len(neg_test), len(pos_test), len(neg_training), len(pos_training)))

            results = do_evaluation(test_set, clf, verbose=verbose)
            cvresults.append(results)

        # output cross validation metrics
        (neg_precision, neg_recall, neg_fmeasure, neg_accuracy, neg_support) = (0.0, 0.0, 0.0, 0.0, 0.0)
        (pos_precision, pos_recall, pos_fmeasure, pos_accuracy, pos_support) = (0.0, 0.0, 0.0, 0.0, 0.0)
        (ave_precision, ave_recall, ave_fmeasure, ave_accuracy, ave_support) = (0.0, 0.0, 0.0, 0.0, 0.0)

        for res in cvresults:
            neg_precision += res['neg']['precision']
            neg_recall += res['neg']['recall']
            neg_fmeasure += res['neg']['fmeasure']
            neg_accuracy += res['neg']['accuracy']
            neg_support += res['neg']['support']
            pos_precision += res['pos']['precision']
            pos_recall += res['pos']['recall']
            pos_fmeasure += res['pos']['fmeasure']
            pos_accuracy += res['pos']['accuracy']
            pos_support += res['pos']['support']
            ave_precision += res['ave']['precision']
            ave_recall += res['ave']['recall']
            ave_fmeasure += res['ave']['fmeasure']
            ave_accuracy += res['ave']['accuracy']
            ave_support += res['ave']['support']

        neg_precision /= kfolds
        neg_recall /= kfolds
        neg_fmeasure /= kfolds
        neg_accuracy /= kfolds
        neg_support /= kfolds

        pos_precision /= kfolds
        pos_recall /= kfolds
        pos_fmeasure /= kfolds
        pos_accuracy /= kfolds
        pos_support /= kfolds

        ave_precision /= kfolds
        ave_recall /= kfolds
        ave_fmeasure /= kfolds
        ave_accuracy /= kfolds
        ave_support /= kfolds

        # print 'neg_precision:', neg_precision
        # print 'neg_recall:', neg_recall
        # print 'neg_fmeasure:', neg_fmeasure
        # print 'neg_accuracy:', neg_accuracy
        # print 'neg_support:', neg_support

        # print 'pos_precision:', pos_precision
        # print 'pos_recall:', pos_recall
        # print 'pos_fmeasure:', pos_fmeasure
        # print 'pos_accuracy:', pos_accuracy
        # print 'pos_support:', pos_support

        # print 'ave_precision:', ave_precision
        # print 'ave_recall:', ave_recall
        # print 'ave_fmeasure:', ave_fmeasure
        # print 'ave_accuracy:', ave_accuracy
        # print 'ave_support:', ave_support

        d['dimension'].append(dimension)
        d['folds'].append(kfolds)
        d['C'].append(C)
        d['class_weight'].append(class_weight)
        
        d['pos_precision'].append(pos_precision)
        d['pos_recall'].append(pos_recall)
        d['pos_fmeasure'].append(pos_fmeasure)
        d['pos_accuracy'].append(pos_accuracy)
        d['pos_support'].append(pos_support)
        
        d['neg_precision'].append(neg_precision)
        d['neg_recall'].append(neg_recall)
        d['neg_fmeasure'].append(neg_fmeasure)
        d['neg_accuracy'].append(neg_accuracy)
        d['neg_support'].append(neg_support)

        d['ave_precision'].append(ave_precision)
        d['ave_recall'].append(ave_recall)
        d['ave_fmeasure'].append(ave_fmeasure)
        d['ave_accuracy'].append(ave_accuracy)
        d['ave_support'].append(ave_support)

    except ValueError:
        print 'FAILED run with:', 'kfolds:', kfolds, '| kernel:', kernel, '| Cs:', C, '| class_weight:', class_weight
        return

    print 'SUCCESSFUL run with:', 'kfolds:', kfolds, '| kernel:', kernel, '| Cs:', C, '| class_weight:', class_weight

# Global constants
verbose = False
save_pickle = True
input_col = 'rev_text_lemm'
random = False
kfolds = 10
kernels = ['linear']

feature = eval(sys.argv[1]) if len(sys.argv) > 1 else unigram_boolean_features

df = pd.read_excel('../proc/sentences_lemm_lab.xlsx')[1:]
d = {
    'dimension': [], 
    'folds': [], 'C': [], 'class_weight': [],
    'pos_precision': [],  'pos_recall': [], 'pos_fmeasure': [], 
    'pos_accuracy': [], 'pos_support': [],
    'neg_precision': [],  'neg_recall': [], 'neg_fmeasure': [], 
    'neg_accuracy': [], 'neg_support': [],
    'ave_precision': [],  'ave_recall': [], 'ave_fmeasure': [], 
    'ave_accuracy': [], 'ave_support': []
}

if not save_pickle:
    # full run set
    Cs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    class_weights = [{1: 1}, {1: 2}, {1: 5}, {1: 10}, {1: 20}, {1: 50}, {1: 100}, {1: 200}, {1: 300}, {1: 500}, {1: 1000}]
    # test set
    # Cs = [0.005]
    # class_weights = [{1: 1}]

    # DEBUG array should read [5:], [5:6] is for testing
    dimensions = df.columns.values[5:]
else:
    dims = {
        'pedoact': {'C': [0.01], 'class_weight': [{1: 100}]},
        'alarm': {'C': [0.05], 'class_weight': [{1: 10}]},
        'ambientlight': {'C': [0.005], 'class_weight': [{1: 100}]},
        'audioplay': {'C': [0.01], 'class_weight': [{1: 50}]},
        'batteryenergy': {'C': [0.01], 'class_weight': [{1: 10}]},
        'calculator': {'C': [0.005], 'class_weight': [{1: 50}]},
        'calls': {'C': [0.02], 'class_weight': [{1: 10}]},
        'directionsgps': {'C': [0.005], 'class_weight': [{1: 100}]},
        'email': {'C': [0.005], 'class_weight': [{1: 20}]},
        'hrm': {'C': [0.02], 'class_weight': [{1: 50}]},
        'messages': {'C': [0.005], 'class_weight': [{1: 200}]},
        'notifications': {'C': [0.01], 'class_weight': [{1: 20}]},
        'speechproc': {'C': [0.01], 'class_weight': [{1: 50}]}
    }
    # Final dimensions that we need pickles for
    # dimensions = ['pedoact']
    dimensions = ['pedoact', 'alarm', 'ambientlight', 'audioplay', 'batteryenergy', 'calculator', 'calls', 'directionsgps', 'email', 'hrm', 'messages', 'notifications', 'speechproc']

# format the data
df.fillna(value=0, inplace=True)
for dimension in dimensions:
    df.loc[df[dimension] == -1, dimension] = 1
if random == True:
    df = df.iloc[np.random.permutation(len(df))]

count = 0
for dimension in dimensions[:]:
    count += 1
    # split the data into class labels
    data = dict(pos = df[df[dimension] == 1], neg = df[df[dimension] == 0])
    # print df[dimension]

    # run the svm with various parameters
    for kernel in kernels:
        for C in Cs if not save_pickle else dims[dimension]['C']:
            for class_weight in class_weights if not save_pickle else dims[dimension]['class_weight']:
                print
                print '====================================================================================='
                print 'Running:', dimension, '| kfolds:', kfolds, '| kernel:', kernel, '| Cs:', C, '| class_weight:', class_weight
                print '====================================================================================='
                if not save_pickle:
                    run_svm_with_params(kernel, C, class_weight, kfolds, d)
                else:
                    train_svm(kernel, C, class_weight, kfolds, d)


out_df = pd.DataFrame(d, columns=['dimension', 
'folds', 'C', 'class_weight',
'pos_precision', 'pos_recall', 'pos_fmeasure', 'pos_accuracy', 'pos_support', 
'neg_precision', 'neg_recall', 'neg_fmeasure', 'neg_accuracy', 'neg_support',
'ave_precision', 'ave_recall', 'ave_fmeasure', 'ave_accuracy', 'ave_support',
])
out_df.to_excel('../proc/eval/perf_svm_' + feature.__name__ + '_' + str(count) + '.xlsx', index=False)