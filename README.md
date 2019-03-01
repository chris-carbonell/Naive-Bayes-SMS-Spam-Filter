# SMS Spam Filter

## Goals

* Classify SMS text messages as spam or ham (i.e., not spam) using a bag-of-words model and a SVM
* Compare the efficiency of a unigram model vs a bigram model

## Executive Summary

Both models scored really well in terms of accuracy:
* Logistic = 93%
* SVM = 98%

Logistic model's confusion matrix:

|                      | Actually Ham | Actually Spam |
|:---------------------|-------------:|--------------:|
| Predicted to be Ham  | 490          | 5             |
| Predicted to be Spam | 0            | 58            |

SVM model's confusion matrix:

|                      | Actually Ham | Actually Spam |
|:---------------------|-------------:|--------------:|
| Predicted to be Ham  | 490          | 5             |
| Predicted to be Spam | 0            | 58            |

## Data

I found the data here: [https://www.kaggle.com/uciml/sms-spam-collection-dataset/home](https://www.kaggle.com/uciml/sms-spam-collection-dataset/home)

The data set provides 5,572 text messages (culled from a variety of sources) classified as either "spam" or "ham". See the [data set's readme](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/01_data/smsspamcollection/readme) for more details.

Of those messages, 4,825 (~87%) are classified as ham with the remaining 747 (~13%) being spam.

## Visualizations

![](https://github.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/blob/master/assets/test.jpg?raw=true)
* distribution by proba

## Prerequisites
* Python 3.x (3.6.0)

## Helpful Links
* [https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd](https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd)
* [https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html)
* [https://machinelearningmastery.com/clean-text-machine-learning-python/](https://machinelearningmastery.com/clean-text-machine-learning-python/)
* [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
* [https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk](https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk)
* [https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)
