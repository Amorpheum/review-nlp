import pandas as pd

filename = raw_input('Enter the data filename (exclude the ext):') or 'data'
wb = pd.ExcelFile('../raw/' + str(filename) + '.xlsx')

# Ignore the first 3 sheets [Contents, URLGenerator and Formula] 
# which are there only for processing purposes. Ignore the last few which
# which are smart glasses
sheetNames = wb.sheet_names[3:-6]

# loop over the sheets, process them
# timestoloop = 1
line_ctr = 0

for sheetName in sheetNames:
    df = wb.parse(sheetName)

    # Removes the last token in the sheetName to simplify it
    rename = ''
    tokens = sheetName.split()
    tokens_len = len(tokens) - 1    # -1 to discount last token
    c = 0
    for token in tokens:
        if c < tokens_len:
            rename = rename + token + ' '
        else:
            break
        c += 1

    df.to_csv('../proc/' + rename.rstrip() + '.csv', encoding='utf=8', index=False)
    line_ctr += len(df)

print 'Number of products:', len(sheetNames)
print 'Number of smart watch reviews:', line_ctr