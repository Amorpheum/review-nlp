import numpy as np
import math
import pandas as pd
from nltk.metrics import scores

"""
Features
"""

"""
This is the simplest possible feature representation of a document. Each word is a feature.
"""
def unigram_boolean_features(words):
    return dict((word, True) for word in words)

"""
Counts the term frequency in a document
"""
def unigram_tf_features(words):
    d = dict()
    for word in words:
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
    return d

"""
Uses tfidf weighting as features
"""
def unigram_tfidf_features(words, idf):
    d = dict()
    for word in words:
        if word not in d:
            d[word] = 1.0
        else:
            d[word] += 1.0

    d = {k: d[k] * idf[k] for k in d}
    return d

"""
Sampling and Sorting out the data
"""
def sample_dataset(sample_array, sample_rate=0.9):
    return int(math.ceil(sample_rate * len(sample_array)))

"""
Turns a set of sentences all belonging to one class into a list of (feature dictionary, cls) pairs, to be used in testing or training a classifier.
"""
def extract_features(corpus, cls=None, feature_extractor=unigram_boolean_features):
    if cls is not None:
        if feature_extractor is unigram_tfidf_features:
            idf = computeidf(corpus)
            return [(feature_extractor(str(string).split(), idf), cls) for string in corpus]
        else:
            return [(feature_extractor(str(string).split()), cls) for string in corpus]
    else:
        # return an array of arrays of features
        return [feature_extractor(str(string).split()) for string in corpus]

"""
Evaluation
"""

def print_basic_info(dimension, feature, sample_sizes):
    banner = "Dimension examined: '%s', Feature used: '%s'" % (dimension, feature)
    print
    print '=' * 75
    print banner
    print '=' * 75
    print 'Test sample = ', sample_sizes[0] + sample_sizes[1], ', Training sample = ', sample_sizes[2] + sample_sizes[3]
    print "'pos' test samples = ", sample_sizes[1], "'neg' test samples = ", sample_sizes[0]
    print "'pos' training samples = ", sample_sizes[3], "'neg' training samples = ", sample_sizes[2]
    print

def do_evaluation(test_set, classifier, verbose=True):
    pairs = generate_pairs(test_set, classifier)
    results = {}

    # Compute the stats for pos and neg cases
    for label in ['pos', 'neg']:
        N = len(pairs)
        (correct, tp, tn, fp, fn) = (0,0,0,0,0)
        for (predicted, actual) in pairs:
            if predicted == actual:
                correct += 1
                if actual == label:
                    tp += 1
                else:
                    tn += 1
            else:
                if actual == label:
                    fn += 1
                else:
                    fp += 1

        accuracy = float(correct)/N
        precision = float(tp)/(tp + fp) if (tp + fp) != 0 else 0
        recall = float(tp)/(tp + fn) if (tp + fn) != 0 else 0
        fmeasure = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        support = tp + fn

        results[label] = {}
        results[label]['accuracy'] = accuracy
        results[label]['precision'] = precision
        results[label]['recall'] = recall
        results[label]['fmeasure'] = fmeasure
        results[label]['support'] = support

    # Compute the ave of pos and neg
    results['ave'] = {}
    results['ave']['accuracy'] = (results['pos']['accuracy'] + results['neg']['accuracy']) / 2
    results['ave']['precision'] = (results['pos']['precision'] + results['neg']['precision']) / 2
    results['ave']['recall'] = (results['pos']['recall'] + results['neg']['recall']) / 2
    results['ave']['fmeasure'] = (results['pos']['fmeasure'] + results['neg']['fmeasure']) / 2
    results['ave']['support'] = results['pos']['support'] + results['neg']['support']

    if verbose:
        print_results(results)

    return results


def print_results(results):

    for key, value in results.iteritems():
        banner = "Evaluation for '%s'" % key
        print
        print banner
        print '-' * len(banner)
        print '%-10s %.1f' % ('Accuracy', value['accuracy']*100)
        print '%-10s %.1f' % ('Precision', value['precision']*100)
        print '%-10s %.1f' % ('Recall', value['recall']*100)
        print '%-10s %.1f' % ('F-measure', value['fmeasure']*100)
        print '%-10s %.1f' % ('Support', value['support'])

def generate_pairs(test_set, classifier):
    return [(classifier.classify(example), actual) for (example, actual) in test_set]

def show_most_informative_features(self, n=10):
    strlist = []
    # Determine the most relevant features, and display them.
    cpdist = self._feature_probdist
    # print('Most Informative Features')
    strlist.append('Most Informative Features')

    for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l,fname].prob(fval)
            labels = sorted([l for l in self._labels
                     if fval in cpdist[l,fname].samples()],
                    key=labelprob)
            if len(labels) == 1: continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0,fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /
                          cpdist[l0,fname].prob(fval))
            # print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
            #      (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))
            strlist.append(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                          (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))

    return strlist

"""
Others
"""
def stratified_k_fold():
    pass


def computeidf(corpus): # words = document, corpus = all documents
    N = len(corpus)
    d = {}

    for words in corpus:
        # count words only once
        s = set(str(words).split())

        # update the dictionary
        for word in s:
            if word not in d:
                d[word] = 1
            else:
                d[word] += 1


    return {k: math.log(N/v) for k, v in d.iteritems()}

# This is an implementation of StratifiedKFold cross validation
def cross_validation_folds(data, dimension, kfolds):
    folded = {}
    folded['neg'] = []
    folded['pos'] = []
    folds = {}
    folds['neg'] = []
    folds['pos'] = []

    neg = data['neg'][dimension]
    pos = data['pos'][dimension]

    sample_rate = 1.0 / kfolds


    # print len(neg), len(pos)

    neg_samp = int(len(neg) * sample_rate)
    pos_samp = int(len(pos) * sample_rate)

    for i in range(kfolds):
        if i != (kfolds - 1):
            folds['pos'].append(pos[i*pos_samp:(i+1)*pos_samp])
            folds['neg'].append(neg[i*neg_samp:(i+1)*neg_samp])
        else:
            folds['pos'].append(pos[i*pos_samp:])
            folds['neg'].append(neg[i*neg_samp:])
    
    # DEBUG
    # count = 0
    # for i in folds['pos']:
    #     for j in i:
    #         count += 1
    # print folds['pos'][0], 'Pos:', len(folds['pos']), count

    for cls in ('pos', 'neg'):
        for i in range(len(folds[cls])):
            fold = {}
            # test
            fold['test'] = folds[cls][i]
            # train. grab the rest
            fold['train'] = pd.Series()
            for j in range(len(folds[cls])):
                if i != j:
                    fold['train'] = pd.concat([fold['train'], folds[cls][j]])
            folded[cls].append(fold)

    # print folded['pos']

    return folded

"""
Import of the WordNetLemmatizer from NLTK to be used in sklearn
"""
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

