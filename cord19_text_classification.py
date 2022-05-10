'''
cord19_text_classification.py : Classify texts as transmission or non transmission
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils_text as utils_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


def model_and_eval(model, X, y):
    ''' Train and evaluate a model. 
    :param: model: classification model
    :param: X: features
    :param: y: target values
    :return: model, y_test (test values), and y_pred (predicted values) '''
    print('Model: {}'.format(model))

    # Create training set with 60% of data and test set with 40% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4, random_state=1
    )

    print("\nTrain-Test Split")

    model.fit(X_train, y_train)  # Train with the train set
    y_pred = model.predict(X_test)  # Predict target values.

    trainset_accuracy_score = model.score(X_train, y_train)
    print("\nThe accuracy score (train set) is: {}".format(trainset_accuracy_score))

    acc = accuracy_score(y_pred, y_test)
    print('\nThe accuracy score (test set) is: {:.4f}'.format(acc))

    # average: {'micro', 'macro', 'samples', 'weighted', 'binary'}
    recall = recall_score(y_pred, y_test, average='weighted')
    print('The recall score (test set) is: {:.4f}'.format(recall))

    precision = precision_score(y_pred, y_test, average='weighted')
    print('The precision score (test set) is: {:.4f}'.format(precision))

    # report = classification_report(y_true=y_test, y_pred=y_pred, target_names=['Non Transmission', 'Transmission'])
    # report = classification_report(y_true=y_test, y_pred=y_pred)
    # print(report)

    
    print("\n10-Fold Cross-Validation")
    mean_cv_score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy').mean()
    print('The mean accuracy score (10-fold CV): {:.4f}'.format(mean_cv_score))

    mean_cv_recall = cross_val_score(model, X_train, y_train, cv=10, scoring='recall_weighted').mean()
    print('The mean recall score (10-fold CV): {:.4f}'.format(mean_cv_recall))

    mean_cv_precision = cross_val_score(model, X_train, y_train, cv=10, scoring='precision_weighted').mean()
    print('The mean precision score (10-fold CV): {:.4f}'.format(mean_cv_precision))

    return model, y_test, y_pred


def plot_confusion_matrix(y_test, y_pred):
    print("plot_confusion_matrix")
    print("y_test {},   y_pred {}".format(type(y_test), type(y_pred)))

    ''' Plot a confusion matrix. 
    :param: y_test: test values
    :param: y_pred: predicted values '''
    cm = confusion_matrix(y_test, y_pred)
    inds = ['Non Transmission', 'Transmission']
    cols = ['Non Transmission', 'Transmission']
    df = pd.DataFrame(cm, index=inds, columns=cols)

    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.20, bottom=0.18)
    sns.set(font_scale=2.0) 
    ax = sns.heatmap(df, cmap='Blues', annot=True, fmt='g', cbar=True)

    font_size = 16

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, horizontalalignment='right', size=font_size)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', size=font_size)

    plt.xlabel('Predicted', size=font_size)
    plt.ylabel('Actual', size=font_size)
    plt.title('COVID-19 Transmission Article Predictions - Actual vs Predicted', size=font_size+2)
    plt.savefig('outputs/confusion_matrix.png')
    # plt.show()


if __name__ == "__main__":
    # Merge transmission and non transmission articles
    transmi_articles = pd.read_csv('data/cord19_transmission_processed.csv', header=0, sep=',')
    non_transmi_articles = pd.read_csv('data/cord19_non_transmission_processed.csv', header=0, sep=',')
    df = pd.concat([transmi_articles, non_transmi_articles], ignore_index=False)
    df = df[["paper_id","title", "abstract", "abstract_processed", "text_body", "is_transmi_article"]]

    print("# of Articles: {}".format(df.shape[0]))
    print(df[['is_transmi_article']].value_counts())

    # Transforms words into numerical data for use in machine learning
    cv, vectorized = utils_text.vectorize_words(df['abstract_processed'].values.astype('U'))

    model = LogisticRegression()
    model, y_test, y_pred = model_and_eval(model, vectorized, df['is_transmi_article'])
    plot_confusion_matrix(y_test, y_pred)

    
    # Test the model with unseen data that is not in the dataset.
    test_articles = ["SARS-CoV-2, the virus that causes COVID-19, spreads from an infected person to others through respiratory droplets and aerosols when an infected person breathes, coughs, sneezes, sings, shouts, or talks. The droplets vary in size, from large droplets that fall to the ground rapidly (within seconds or minutes) near the infected person, to smaller droplets, sometimes called aerosols, which linger in the air, especially in indoor spaces. The relative infectiousness of droplets of different sizes is not clear. Infectious droplets or aerosols may come into direct contact with the mucous membranes of another person's nose, mouth or eyes, or they may be inhaled into their nose, mouth, airways and lungs. The virus may also spread when a person touches another person (i.e., a handshake) or a surface or an object (also referred to as a fomite) that has the virus on it, and then touches their mouth, nose or eyes with unwashed hands.",
    "Omicron transmissibility and virulence: What do they mean? There are a number of aspects or words that we use to describe the behavior of a virus. One is infectivity or transmissibility. Let's say I was infected. What's the opportunity or the risk that I could spread that to you, an unimmunized and uninfected person? Virulence refers to the likelihood of disease that would be caused by that infection.",
    "Efforts should be made to improve knowledge about the benefits of vaccines in general and of COVID-19 vaccines as each becomes available, address misinformation, and communicate transparently about COVID-19 allocation decisions.",
    "All reported MERS cases have been linked to countries in and near the Arabian Peninsula. Most infected people either lived in the Arabian Peninsula or recently traveled from the Arabian Peninsula before they became ill. A few people have gotten MERS after having close contact with an infected person who had recently traveled from the Arabian Peninsula. The largest known outbreak of MERS outside the Arabian Peninsula occurred in the Republic of Korea in 2015 and was associated with a traveler returning from the Arabian Peninsula."]

    cleaned_test_articles = []
    for article in test_articles:
        article = article.lower()
        article = utils_text.lemmatize_words(article)
        article = utils_text.remove_punctuations(article)
        article = utils_text.remove_stopwords(article)
        cleaned_test_articles.append(article)

    X_test = cv.transform(cleaned_test_articles)
    pred_results = model.predict(X_test)
    print("Is COVID-19 Tranmission Article? {}".format(pred_results))

    