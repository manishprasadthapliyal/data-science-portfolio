# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter= '\t',quoting=3)

# CLeaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 1000) :
    # Only keep letters
    # First Step : Kept All the letters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])    
    
    # Second Step : Change all letters to lower case 
    review = review.lower()
    
    # Third Step: Remove unwanted words
    review = review.split()
    
    # Fourth Step : Stemming - Taking root of the word 
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    # joining back list of clean words

    review = ' '.join(review)
    corpus.append(review)
    
# Creating Bag of Words model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()    
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

    
    
    



