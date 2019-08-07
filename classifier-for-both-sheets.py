import pandas as pd 
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

file_path = input('Write the path to which file you want here: ')

df = pd.read_csv(file_path, encoding='unicode_escape')
print(df.head())
print(df.columns)

text_name = input('Which column has the text you need: ')
label_name = input('Which column has the labels you need: ')

text = list(df[text_name])
labels = list(df[label_name])
labels = [1 if x == 'flagged' else 0 for x in labels]
print(len(text))
print(len(labels))
print(labels)

training_text, test_text, training_labels, test_labels = train_test_split(
    text, labels, train_size=0.8, test_size=0.2)

bow_vectorizer = CountVectorizer()

training_vectors = bow_vectorizer.fit_transform(training_text)
test_vectors = bow_vectorizer.transform(test_text)

classifier = MultinomialNB()

classifier.fit(training_vectors, training_labels)

predictions = classifier.score(test_vectors, test_labels)

def which_classifier():
    return 'Therapy Bot Classifier' if file_path == 'Sheet_1.csv' else 'Resume Classifier'

print('Predictions for test data were {}% accurate for the {}.'.format(
    predictions * 100, which_classifier()))
