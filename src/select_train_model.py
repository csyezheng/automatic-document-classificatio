"""
Select and Train Model
    Vectorizes training and validation texts
    training a Some model
    Evaluate model using Cross-Validation
"""

from load_data import load_bbc_news_dataset
from vectorize_data import ngram_vectorize

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib



def train_model(data):
    """
    Trains Some model on the given dataset.
    """

    (train_texts, train_labels), (val_texts, val_labels) = data

    x_train, x_val = ngram_vectorize(
        train_texts, train_labels, val_texts)

    # Create model instance.
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]

    # Train and evaluate model.
    # evaluate each model using K-fold cross-validation
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, x_train, train_labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
        joblib.dump(model, "models/{0}.pkl".format(model_name))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    model_accuracy = cv_df.groupby('model_name').accuracy.mean()
    print(model_accuracy)
    return


if __name__ == '__main__':
    data = load_bbc_news_dataset("./data/bbc-text.csv")
    train_model(data)
