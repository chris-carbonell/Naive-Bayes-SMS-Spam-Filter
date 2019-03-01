import pandas as pd
import seaborn as sns
import csv

from nltk import stem
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix

import pprint

def scrub_txt(dirty_txt):
    '''scrub text: lower, stop words, and stem

    Argument(s):
    dirty_txt (str): string to clean

    Return:
    clean_txt (str): clean string
    '''

    # potential improvements:
    # - remove punctuation

    stemmer = stem.SnowballStemmer('english')
    eng_stopwords = set(stopwords.words('english'))

    # lower, stop, stem
    clean_txt = dirty_txt.lower()
    clean_txt = [word for word in clean_txt.split() if word not in eng_stopwords]
    clean_txt = " ".join([stemmer.stem(word) for word in clean_txt])

    return clean_txt

def get_collection_prediction(svm, sms_txt):
    '''predict whether ham or spam

    Argument(s):
    sms_txt (str): sms message text

    Return:
    prediction
    '''

    sms_txt = vectorizer.transform([sms_txt])
    prediction = svm.predict(sms_txt)

    return prediction[0]

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # import data
    df_data = pd.read_csv("../01_data/smsspamcollection/SMSSpamCollection",
        sep='\t', encoding="latin-1",
        header=None, names=["collection", "sms_txt_dirty"])

    # quick scrub
    df_data["sms_txt_clean"] = df_data["sms_txt_dirty"].apply(scrub_txt)

    # split
    # straitfy?
    X_train, X_test, y_train, y_test = train_test_split(
        df_data["sms_txt_dirty"], df_data["collection"],
        test_size=0.1, random_state=1)

    # vetorize
    vectorizer = TfidfVectorizer()
    X_train_v = vectorizer.fit_transform(X_train)

    # svm
    svm = svm.SVC(C=1000, gamma='auto')
    svm.fit(X_train_v, y_train)

    # test model
    X_test_v = vectorizer.transform(X_test)
    y_pred = svm.predict(X_test_v)
    pp.pprint(confusion_matrix(y_test, y_pred))

    # quick examples
    df_results = pd.DataFrame.from_dict({'X_test':X_test, 'y_test': y_test, 'y_pred': y_pred})
    pp.pprint(df_results[df_results['y_test'] == 'ham'].head())
    pp.pprint(df_results[df_results['y_test'] == 'spam'].head())
