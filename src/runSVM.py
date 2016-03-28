def unigram_features(words):
    return dict((word, True) for word in words)

def extract_features(corpus, file_ids, cls, feature_extractor=unigram_features):
    return [(feature_extractor(corpus.words(i)), cls) for i in file_ids]

def get_words_from_corpus (corpus, file_ids): 
    for file_id in file_ids: 
        words = corpus.words(file_id) 
        for word in words: 
            yield word

from nltk.corpus import movie_reviews as mr
from nltk.classify import NaiveBayesClassifier
import nltk.classify
from sklearn.svm import NuSVC

data = dict(pos = mr.fileids('pos'), neg = mr.fileids('neg'))

# Use 90% of the data for training 
neg_training = extract_features(mr, data['neg'][:900], 'neg', feature_extractor=unigram_features) 
# Use 10% for testing the classifier on unseen data. 
neg_test = extract_features(mr, data['neg'][900:], 'neg', feature_extractor=unigram_features) 
pos_training = extract_features(mr, data['pos'][:900],'pos', feature_extractor=unigram_features) 
pos_test = extract_features(mr, data['pos'][900:],'pos', feature_extractor=unigram_features) 
train_set = pos_training + neg_training 
test_set = pos_test + neg_test

# classifier = NaiveBayesClassifier.train(train_set)
classifier = nltk.classify.SklearnClassifier(NuSVC())
classifier.train(train_set)

predicted_label0 = classifier.classify(pos_test[0][0]) 
print 'Predicted: %s Actual: pos' % (predicted_label0,) 
predicted_label1 = classifier.classify(neg_test[0][0]) 
print 'Predicted: %s Actual: neg' % (predicted_label1,)

print 'Inception is the best movie ever:', classifier.classify(unigram_features('Inception is the best movie ever'.split()))

print "I don't know how anyone could sit through Inception:",  classifier.classify(unigram_features("I don't know how anyone could sit through Inception".split()))


# classifier.show_most_informative_features()


def do_evaluation (pairs, pos_cls='pos', verbose=True): 
    N = len(pairs) 
    (ctr,correct, tp, tn, fp,fn) = (0,0,0,0,0,0) 
    for (predicted, actual) in pairs: 
        ctr += 1 
        if predicted == actual: 
            correct += 1 
            if actual == pos_cls: 
                tp += 1 
            else: 
                tn += 1 
        else: 
            if actual == pos_cls: 
                fp += 1 
            else: 
                fn += 1 
    (accuracy, precision, recall) = (float(correct)/N,float(tp)/(tp + fp),float(tp)/(tp + fn)) 
    if verbose: 
        print_results(precision, recall, accuracy, pos_cls) 
    return (accuracy, precision, recall)

def print_results (precision, recall, accuracy, pos_cls): 
    banner = 'Evaluation with pos_cls = %s' % pos_cls 
    print 
    print banner 
    print '=' * len(banner) 
    print '%-10s %.1f' % ('Precision',precision*100) 
    print '%-10s %.1f' % ('Recall',recall*100) 
    print '%-10s %.1f' % ('Accuracy',accuracy*100)

pairs = [(classifier.classify(example), actual) for (example, actual) in test_set] 
do_evaluation (pairs) 
do_evaluation (pairs, pos_cls='neg')









