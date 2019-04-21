## Step 1: Gather Data

Source data from public data set on BBC news articles. 

Its original source.its original source:  <http://mlg.ucd.ie/datasets/bbc.html>

Cleaned up version: <https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv>



## Step 2: Explore Data

Is the number of articles in each category roughly equal?

If our dataset were imbalanced, we would need to carefully configure our model or artificially balance the dataset, for example by [**undersampling** or **oversampling**](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) each class.



## Step 2.5: Choose a Model





## Step 3: Prepare Data

### extracting features

To further analyze our dataset, we need to transform each article's text to a feature vector, a list of numerical values representing some of the text’s characteristics. 

* one-hot vector
* Co-occurrence Matrix (SVD Based Methods)

* ##### word2vec 

  * ##### continuous bag of words model

    a model where for each document, an article in our case, the presence (and often the **frequency**) of words is taken into consideration, but the order in which they occur is ignored.

  * skip-gram

* Term Frequency, Inverse Document Frequency (**tf-idf**)

  This statistic represents words’ importance in each document.



## Step 4: Build, Train, and Evaluate Model



Select and Train a Model

We don't want to touch the test set until we are ready to launch a model you are confident about, so we need to use part of the training set for training, and part for validation.



### Performance Measures

#### Measuring Accuracy Using Cross-Validation

#### Confusion Matrix

#### Precision and Recall

#### Precision / Recall Tradeoff

#### The ROC Curve

we must carefully choose the right metric based on the task we are trying to solve. Here, we are dealing with a multi-class classification task (trying to assign one label to each document, out of a number of labels). Given the relative balance of our dataset, **accuracy** does seem like a good metric to optimize for. If one of the labels was more important than the others, we would then weight it higher, or focus on its own results. For imbalanced datasets such as the one described above, it is common to look at the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic), and optimize the Area Under the Curve (**ROC AUC**).

## Step 5: Tune Hyperparameters

## Step 6: Deploy Model

