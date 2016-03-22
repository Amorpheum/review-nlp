import pandas as pd
import math

sampleRate = float(raw_input('Enter a sample rate (%): ') or 2.65) / 100
print 'Sample Rate:', sampleRate
df = pd.read_csv('../proc/sentences_lemm.csv')
totalSampleNumRev = math.ceil(sampleRate * len(set(df['rev_no'])))
filenames = sorted(set(df['filename']))

# Total number of samples per device
pdtSampRevNums = []
for filename in filenames:
    # retrieve rows of a filename
    pdtRows = df.loc[df['filename'] == filename]
    revNum = len(set(pdtRows['rev_no']))
    sampNum = int(math.ceil(revNum * sampleRate))

    pdtSampRevNums.append((filename, sampNum, revNum))

labelled = pd.DataFrame()
unlabelled = pd.DataFrame()

# separate the data based on the sample rates into two csvs
for (filename, sampleNum, revNum) in pdtSampRevNums:
    # rows from a file
    filedf = df.loc[df['filename'] == filename]

    start_lab = filedf['rev_no'].iloc[0]
    end_lab = start_lab + sampleNum
    lab_range = range(start_lab, end_lab)
    unlab_range = range(end_lab, start_lab + revNum)

    # DEBUG
    # print filedf.loc[filedf['rev_no'].isin(lab_range)]['rev_text']
    # print 'lab_range'
    # print filedf.loc[filedf['rev_no'].isin(unlab_range)]['rev_text']
    # print 'unlab_range'

    labelled = pd.concat([labelled, filedf.loc[filedf['rev_no'].isin(lab_range)]])
    unlabelled = pd.concat([unlabelled, filedf.loc[filedf['rev_no'].isin(unlab_range)]])

# output the data
labelled.to_csv('../proc/sentences_lemm_lab.csv', index=False)
unlabelled.to_csv('../proc/sentences_lemm_unlab.csv', index=False)

