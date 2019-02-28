import pandas as pd
import seaborn as sns
import csv

import pprint

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # can't read CSV with read_csv
    #df_data = pd.read_csv("../01_data/SMSSpamCollection.csv", header=None)
    # df_data = df_data.iloc[:, :2] # I can't use this bc of .iloc[235, :]

    # read in CSV manually
    df_data = pd.DataFrame(columns=['collection', 'sms_txt'])
    with open("../01_data/SMSSpamCollection.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            df_data = df_data.append(
                {'collection': row[0], 'sms_txt': ','.join(row[i for i, item in enumerate(row[1:]) if item is not None])}, ignore_index=True)

    # pprint
    pp.pprint(df_data.head())
    pp.pprint(df_data.isnull().sum())
    pp.pprint(df_data.notnull().sum())

    pp.pprint(df_data.iloc[235:240, :])
