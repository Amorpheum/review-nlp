import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import cross_validation
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.grid_search import GridSearchCV
from classifyUtils import * # utility class
import sys

# Select a classifier, default being MultinomialNB
clf_name = sys.argv[1] if len(sys.argv) > 1 else 'mnb'
if clf_name == 'svm':
    C = 0.05
    class_weight = {1: 10}
    clf = svm.SVC(kernel='linear'
        , C=C
        , class_weight=class_weight
        )    
elif clf_name == 'bnb':
    clf = BernoulliNB()
else:
    clf = MultinomialNB()



# Prepare the data
df = pd.read_excel('../proc/sentences_lemm_lab.xlsx')[1:]
df.fillna(value=0, inplace=True)
df['rev_text'] = df['rev_text'].astype(str)
dimensions = df.columns.values[5:]
# recode the data so that -1 is also 1, that is negative and positive reviews are both useful
for dimension in dimensions:
    df.loc[df[dimension] == -1, dimension] = 1

cat = 'design'
# data = [str(i) for i in df['rev_text']]
data = df['rev_text']
target = df[cat]

feat_name = sys.argv[2] if len(sys.argv) > 2 else 'countvec'
if feat_name == 'tfidf':
    feat = (
        'vect'
        , TfidfVectorizer(max_df=0.95
            , stop_words='english'
            # , tokenizer=LemmaTokenizer()
        )
    )
elif feat_name == 'hashvec':
    feat = (
        'vect'
        , HashingVectorizer(stop_words='english', non_negative=True)
    )
else:
    feat = (
        'vect'
        , CountVectorizer(stop_words='english')
    )

# Set up the Pipeline
# sw_clf = Pipeline([feat, ('clf', clf)])
sw_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', clf)])

# Set up GridSearch
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
    'tfidf__use_idf': (True, False),
    'clf__C': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(sw_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data, target)

best_params, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print '%s: %r' % (param_name, best_params[param_name])

print score

# predicted = cross_validation.cross_val_predict(sw_clf, data, target, cv=10)
# print metrics.classification_report(target, predicted, target_names=['neg', 'pos'])
print 'C =', C, 'class_weight =', class_weight