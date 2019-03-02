import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pprint

if __name__ == "__main__":

    pp = pprint.PrettyPrinter(indent=4)

    df_log = pd.read_csv("../03_results/01_model_results/log_results.csv",
        encoding='utf-8',
        header=0,
        index_col=0)

    df_svm = pd.read_csv("../03_results/01_model_results/svm_results.csv",
        encoding='utf-8',
        header=0,
        index_col=0)

    df_all = pd.merge(df_log, df_svm[['y_pred', 'y_pred_proba']], how='left',
        left_index=True, right_index=True)

    # pp.pprint(df_log.head())
    # pp.pprint(df_svm.head())
    # print(list(df_all.columns.values))
    # pp.pprint(df_all.head())

    # distribution of y_pred_proba_x
    plot = sns.distplot(df_all['y_pred_proba_x'], axlabel='Model Prediction (0=spam, 1=ham)', label='Message Count')
    plot.set(xlim=(0,1), ylim=(0,None))
    plt.savefig("../03_results/02_visualizations/plot_01.png", dpi=400)

    # distribution of y_pred_proba_y
    plot = sns.distplot(df_all['y_pred_proba_x'], axlabel='Model Prediction (0=spam, 1=ham)', label='Message Count')
    plot.set(xlim=(0,1), ylim=(0,None))
    plt.savefig("../03_results/02_visualizations/plot_02.png", dpi=400)
