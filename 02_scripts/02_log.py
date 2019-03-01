import pandas as pd
import seaborn as sns

from nltk import stem
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import pprint

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # import data
    df_data = pd.read_csv("../01_data/sms.csv",
        encoding='utf-8',
        header=0)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        df_data['sms_txt_dirty'], df_data['collection_log'],
        test_size=0.1, random_state=1)

    # vetorize
    vectorizer = TfidfVectorizer()
    X_train_v = vectorizer.fit_transform(X_train)

    # logistic regression
    logisticRegr = LogisticRegression(solver='lbfgs')
    logisticRegr.fit(X_train_v, y_train)

    # test model
    X_test_v = vectorizer.transform(X_test)
    y_pred = logisticRegr.predict(X_test_v)
    y_pred_proba = logisticRegr.predict_proba(X_test_v) # ham probability

    # summary
    pp.pprint(confusion_matrix(y_test, y_pred))
    score = logisticRegr.score(X_test_v, y_test)
    print(score) # 93%

    # save results
    df_results = pd.DataFrame.from_dict(
        {'X_test':X_test,
        'y_test': ['ham' if x == 1 else 'spam' for x in y_test],
        'y_pred': ['ham' if x == 1 else 'spam' for x in y_pred],
        'y_pred_proba': y_pred_proba[:, 1].tolist()})
    df_results.to_csv("../03_results/01_model_results/log_results.csv",
        encoding='utf-8', index=True)

    # quick examples
    #pp.pprint(df_results[df_results['y_test'] == 'ham'].head())
    #pp.pprint(df_results[df_results['y_test'] == 'spam'].head())

    df_results[(df_results['y_pred_proba'] > 0.9) & (df_results['y_test'] == 'ham')].to_csv("../03_results/01_model_results/log_easy_ham.csv")
    df_results[(df_results['y_pred_proba'] < 0.1) & (df_results['y_test'] == 'spam')].to_csv("../03_results/01_model_results/log_easy_spam.csv")
    df_results[df_results['y_pred_proba'].between(.45,.55, inclusive=True)].to_csv("../03_results/01_model_results/log_tough.csv")
