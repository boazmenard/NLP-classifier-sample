import pandas as pd 
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load the file and clean up the dataframe
# Visualize the data to get a feel
resumes = pd.read_csv('./Sheet_2.csv', encoding='unicode_escape')
print(resumes.head())

resume_text = list(resumes['resume_text'])
labels = list(resumes['class'])

labels = [0 if label == 'not_flagged' else 1 for label in labels]
print(len(labels))
print(len(resume_text))

training_text, test_text, training_labels, test_labels = train_test_split(
    resume_text, labels, train_size=0.8, test_size=0.2)

bow_vectorizer = CountVectorizer()

training_vectors = bow_vectorizer.fit_transform(training_text)
test_vectors = bow_vectorizer.transform(test_text)

resume_classifier = MultinomialNB()
resume_classifier.fit(training_vectors, training_labels)

predictions = resume_classifier.score(test_vectors, test_labels)

print('Predictions for test data were {}% accurate'.format(predictions * 100))

