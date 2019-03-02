# SMS Spam Filter

## Goals

* Classify SMS text messages as spam or ham (i.e., not spam) using a bag-of-words model and a SVM model

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

Therefore, at least between these two models, the SVM model would be the suggested model to use for identifying spam messages.

## Examples of Messages
Here are a few examples of messages that are very likely ham, very likely spam, and some messages that were tougher to classify based on the SVM model. Probabilities are rounded to three decimals.

### Very Likely Ham

| Message                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | SVM Probability of Ham |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| Sary just need Tim in   the bollox &it hurt him a lot so he tol me!                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 1.000                  |
| Sad story of a Man - Last week was my b'day. My Wife did'nt wish me. My Parents forgot n so did my Kids . I went to work. Even my Colleagues did not wish. As I entered my cabin my PA said, '' Happy B'day Boss !!''. I felt special. She askd me 4 lunch. After lunch she invited me to her apartment. We went there. She said,'' do u mind if I go into the bedroom for a minute ? '' ''OK'', I sed in a sexy mood. She came out 5 minuts latr wid a cake...n My Wife, My Parents, My Kidz, My Friends n My Colleagues. All screaming.. SURPRISE !! and I was waiting on the sofa.. ... ..... ' NAKED...! | 1.000                  |
| Hello, my boytoy! I made it home and my constant thought is of you, my love. I hope your having a nice visit but I can't wait till you come home to me ...*kiss*                                                                                                                                                                                                                                                                                                                                                                                                                                             | 1.000                  |
| Yar he quite clever but aft many guesses lor. He got ask me 2 bring but i thk darren not so willing 2 go. Aiya they thk leona still not attach wat.                                                                                                                                                                                                                                                                                                                                                                                                                                                          | 1.000                  |
| Hey babe, sorry i didn't get sooner. Gary can come and fix it cause he thinks he knows what it is but he doesn't go as far a Ptbo and he says it will cost  &lt;#&gt;  bucks. I don't know if it might be cheaper to find someone there ? We don't have any second hand machines at all right now, let me know what you want to do babe                                                                                                                                                                                                                                                                      | 1.000                  |

### Very Likely Spam

| Message                                                                                                                                                          | SVM Probability of Ham |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| Do you want a New Nokia 3510i Colour Phone Delivered Tomorrow? With 200 FREE minutes to any mobile + 100 FREE text + FREE camcorder Reply or Call 08000930705    | 0.000                  |
| SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info                         | 0.000                  |
| Great NEW Offer - DOUBLE Mins & DOUBLE Txt on best Orange tariffs AND get latest camera phones 4 FREE! Call MobileUpd8 free on 08000839402 NOW! or 2stoptxt T&Cs | 0.000                  |
| Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs www.Ldew.com1win150ppmx3age16           | 0.000                  |
| Free tones Hope you enjoyed your new content. text stop to 61610 to unsubscribe. help:08712400602450p Provided by tones2you.co.uk                                | 0.000                  |

### Tough to Classify

| Message                                                                                                                                                      | SVM Probability of Ham |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| LookAtMe!: Thanks for your purchase of a video clip from LookAtMe!, you've been charged 35p. Think you can do better? Why not send a video in a MMSto 32323. | 0.546                  |
| When you get free, call me                                                                                                                                   | 0.514                  |
| Cheers for the message Zogtorius. IÃ‚Â’ve been staring at my phone for an age deciding whether to text or not.                                               | 0.491                  |

## Data

I found the data here: [https://www.kaggle.com/uciml/sms-spam-collection-dataset/home](https://www.kaggle.com/uciml/sms-spam-collection-dataset/home)

The data set provides 5,572 text messages (culled from a variety of sources) classified as either "spam" or "ham". See the [data set's readme](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/01_data/smsspamcollection/readme) for more details.

Of those messages, 4,825 (~87%) are classified as ham with the remaining 747 (~13%) being spam.

## Visualizations

In the following visualizations, the x-axis ranges from 0, spam, to 1, ham. For example, a prediction of 0.4 indicates the message is probably spam but not super spammy.

### Logistic Model's Histogram of Probabilities
![](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/03_results/02_visualizations/plot_01.png?raw=true)

### SVM Model's Histogram of Probabilities
![](https://raw.githubusercontent.com/chris-carbonell/Naive-Bayes-SMS-Spam-Filter/master/03_results/02_visualizations/plot_02.png?raw=true)

## Prerequisites
* Python 3.x (3.6.0)

## Future Enhancements
* compare predictive power of unigram and bigram models

## Resources
* [https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd](https://towardsdatascience.com/spam-or-ham-introduction-to-natural-language-processing-part-2-a0093185aebd)
* [https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html)
* [https://machinelearningmastery.com/clean-text-machine-learning-python/](https://machinelearningmastery.com/clean-text-machine-learning-python/)
* [https://scikit-learn.org/stable/modules/feature_extraction.html](https://scikit-learn.org/stable/modules/feature_extraction.html)
* [https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk](https://stackoverflow.com/questions/37651057/generate-bigrams-with-nltk)
* [https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a](https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a)
