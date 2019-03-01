import pandas as pd

from nltk import stem
from nltk.corpus import stopwords

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

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # import data
    df_data = pd.read_csv("../01_data/smsspamcollection/SMSSpamCollection",
        sep='\t', encoding="latin-1",
        header=None, names=["collection", "sms_txt_dirty"])

    # quick scrub
    df_data["sms_txt_clean"] = df_data["sms_txt_dirty"].apply(scrub_txt)
    df_data["collection_log"] = 1
    df_data.loc[df_data["collection"] == "spam", "collection_log"] = 0

    df_data.to_csv("../01_data/sms.csv", encoding='utf-8', index=False)

    # quick summary on incoming data
    pp.pprint(df_data.shape) # (5572, 4)
