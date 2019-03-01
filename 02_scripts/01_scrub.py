import pandas as pd
import seaborn as sns
import csv

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords

import pprint

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # import data
    df_data = pd.read_csv("../01_data/smsspamcollection/SMSSpamCollection",
        sep = '\t', encoding = "latin-1",
        header = None, names = ["collection", "sms_txt_dirty"]
        )
