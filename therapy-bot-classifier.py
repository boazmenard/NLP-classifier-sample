import pandas as pd 
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Check which input data files are available in the directory.
# Running this block will list all files under the input directory
import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the file and clean up the dataframe
# Visualize the data to get a feel
responses_df = pd.read_csv('./Sheet_1.csv')
responses_df.drop(columns=["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7"]
                  , inplace=True)
print(responses_df.head())

# Single out the data needed to create the Classifier
text = list(responses_df['response_text'])
labels = list(responses_df['class'])

# Convert the labels to boolean labeling (0s and 1s)
labels = [0 if label == 'not_flagged' else 1 for label in labels]

# Split data into training and test sets
# Define NEW text to test against classifier
training_text, test_text, training_labels, test_labels = train_test_split(text, labels,
                                                            train_size=0.8, test_size=0.2)
testing = ["I love you"]

# Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer
bow_vectorizer = CountVectorizer()

# Create vectors
training_vectors = bow_vectorizer.fit_transform(training_text)
test_vectors = bow_vectorizer.transform(test_text)
new_testing = bow_vectorizer.transform(testing) #vector to test the classifier

# Create classifier and train it
flag_classifier = MultinomialNB()
flag_classifier.fit(training_vectors, training_labels)

# Convert the boolean label to category
def flag_or_not(label):
    return "FLAGGED" if label else "NOT FLAGGED"

predictions = flag_classifier.score(test_vectors, test_labels)
print('Predictions for test data were {}% accurate'.format(predictions * 100))

predicted = flag_classifier.predict(new_testing)
print("The classifier is predicting the new unlabeled text to be of label {} which is {}."
      .format(predicted, flag_or_not(predicted)))

