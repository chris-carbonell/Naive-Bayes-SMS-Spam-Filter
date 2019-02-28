import pandas as pd
import seaborn as sns
import csv

import pprint

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    # can't read CSV with read_csv
    # df_data = pd.read_csv("../01_data/SMSSpamCollection.csv", header=None)
    # df_data = df_data.iloc[:, :2] # I can't use this bc of .iloc[235, :]

    # read in CSV manually
    df_data = pd.DataFrame(columns=['collection', 'sms_txt'])
    with open("../01_data/SMSSpamCollection.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            sms_txt = [txt for txt in row[1:] if str(txt) != '']  # get rid of empty entries
            df_data = df_data.append(
                {'collection': row[0], 'sms_txt': ','.join(
                    sms_txt)},
                ignore_index=True)

    # pprint
    pp.pprint(df_data.head())
    pp.pprint(df_data.isnull().sum())
    pp.pprint(df_data.notnull().sum())

    pp.pprint(df_data.iloc[235:240, :])
