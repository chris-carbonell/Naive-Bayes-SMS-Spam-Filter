# SMS Spam Filter

## Goals

* Classify SMS text messages as spam or ham (i.e., not spam) using a bag-of-words model and a SVM
* Compare the efficiency of a unigram model vs a bigram model

## Executive Summary

Both models scored well in terms of accuracy:
* Logistic = 93%
* SVM = 98%

Logistic model's confusion matrix (testing data):

|                      | Actually Ham | Actually Spam |
|:---------------------|-------------:|--------------:|
| Predicted to be Ham  | 487          | 2             |
| Predicted to be Spam | 7            | 62            |

SVM model's confusion matrix (testing data):

|                      | Actually Ham | Actually Spam |
|:---------------------|-------------:|--------------:|
| Predicted to be Ham  | 488          | 1             |
| Predicted to be Spam | 8            | 61            |

## Data

I found the data here: [https://www.kaggle.com/uciml/sms-spam-collection-dataset/home](https://www.kaggle.com/uciml/sms-spam-collection-dataset/home)

The data set provides 5,572 text messages (culled from a variety of sources) classified as either "spam" or "ham". See the [data set's readme](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/01_data/smsspamcollection/readme) for more details.

Of those messages, 4,825 (~87%) are classified as ham with the remaining 747 (~13%) being spam.

## Visualizations

In the following visualizations, the x-axis ranges from 0, spam, to 1, ham. For example, a prediction of 0.4 indicates the message is probably spam but not super spammy.

### Logistic model's histogram of probabilities
![](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/03_results/02_visualizations/plot_01.png?raw=true)

### SVM model's histogram of probabilities
![](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/03_results/02_visualizations/plot_02.png?raw=true)

## Prerequisites
* Python 3.x (3.6.0)

## Resources
* [https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd](https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd)
* [https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html)
* [https://machinelearningmastery.com/clean-text-machine-learning-python/](https://machinelearningmastery.com/clean-text-machine-learning-python/)
* [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
* [https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk](https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk)
* [https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)
