import pandas as pd
import nltk
from utils import *

manifile = '../' + (str(raw_input('Enter the manifest file name: ')) or 'manifest') + '.txt'

fh = open(manifile)
filenames = fh.read().splitlines()

rev_ctr = 0
d = {'filename': [], 'rev_no': [], 'rev_text': [], 'rev_text_lemm': []}  # column data: filename | rev_no | rev_text | ... (the rest to be created by manual coding)

print '=== Started processing ' + manifile[3:] + ' ==='
for f in filenames:
    print 'processing', f, '...'
    df = pd.read_csv('../proc/basecsvs/' + f + '.csv')

    # Parse into sentences
    sdf = df[['rev_text']].applymap(parseToSentences)

    for review in sdf['rev_text']:
        # print sdf['rev_text']
        for sentence in review:
            d['filename'].append(f)
            d['rev_no'].append(rev_ctr)
            d['rev_text'].append(sentence)
            d['rev_text_lemm'].append(cleanText(sentence))
        rev_ctr += 1

newdf = pd.DataFrame(d)
newdf.index.name = 'idx'
newdf.to_csv('../proc/sentences_lemm.csv')

# DEBUG
# ndf = pd.DataFrame.from_csv('../proc/sentences.csv')
# ndf = pd.read_csv('../proc/sentences.csv', index_col=0)
# print ndf

print '=== Finished processing ' + manifile[3:] + ' ==='