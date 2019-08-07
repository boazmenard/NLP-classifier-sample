# NLP-classifier-sample
NLP Classifier Sample with datasets found on Kaggle.

## Contents
- Sheet_1.csv - A csv document that holds data relating to a therapy bot and responses for the user. Responses that were seen as negative, threatening, indicating harm, etc. were classified as flagged.
- Sheet_2.csv - A csv document that holds data relating to resume of applicants for a job. The resumes that were invited to interview were classified as flagged.
- Therapy Bot Classifer - Python script which creates a classifier to predict whether text should be flagged or not.
- Resume Classifier - Python script which creates a classifier to predict whether an applicant should be invited to interview based on resume contents (flagged) or not.
- Classifier-for-both - Python Script which needs to be run in same directory as the two sheets. This allows the user to dictate which classifier should be created based on need.
