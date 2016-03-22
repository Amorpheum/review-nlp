import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
from string import punctuation
import re

def stripPunctuation(text):
    return ''.join(c for c in text if c not in punctuation)

def removeNonASCII(text):
    return re.sub(r'[^\x00-\x7F]+', '', str(text))

def removeHyphen(text):
    return re.sub('-+', ' ', str(text))

def parseToSentences(text):
    return nltk.tokenize.sent_tokenize(removeHyphen(removeNonASCII(text)))

def cleanText(text):
    # remove non-ascii characters, lower, then tokenize text
    tokens = nltk.word_tokenize(stripPunctuation(removeNonASCII(text)).lower())

    # stop word removal
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]

    # Perform stemmming or lemmatization. Default is lemmatization
    # porter = PorterStemmer()
    # snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()

    final_sentence = []
    # [final_doc.append(porter.stem(word)) for word in tokens]
    # [final_doc.append(snowball.stem(word)) for word in tokens]
    # [final_doc.append(wordnet.lemmatize(word)) for word in tokens]
    [final_sentence.append(wordnet.lemmatize(word)) for word in filtered_words]

    return ' '.join(final_sentence)
