'''Building an email spam detector involves several steps, including data
 preparation, feature extraction, model training, and evaluation. Here'
s an example of how you can create a simple email spam detector using Python and scikit-learn:'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

# Load the dataset
spam = pd.read_csv('spam.csv', encoding='latin1')

# Split the data into input (z) and target (y)
z = spam['v2']
y = spam["v1"]

# Split the data into training and testing sets
z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2)

# Create a CountVectorizer object
cv = CountVectorizer()

# Fit and transform the training data
features = cv.fit_transform(z_train)

# Create an SVM model and train it
model = svm.SVC()
model.fit(features, y_train)

# Transform the testing data
features_test = cv.transform(z_test)

# Calculate and print the accuracy
accuracy = model.score(features_test, y_test)
print("Accuracy: {}".format(accuracy))

# Plot the accuracy graph
labels = ['Accuracy']
values = [accuracy]
plt.bar(labels, values)
plt.xlabel('Metrics')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()
