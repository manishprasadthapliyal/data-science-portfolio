# Toxic Comment Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
rows = 1000
# Import dataset
dataset = pd.read_csv('train.csv')
# CLeaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def bagOfWords(beg, end):
    corpus = []
    for i in range(beg, end) :
        # Only keep letters
        # First Step : Kept All the letters
        review = re.sub('[^a-zA-Z]', ' ', dataset['comment_text'][i])    
        
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=1500)
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:rows,2:8].values
    data = {
            'X':X,
            'y':y
            }
    return data
bow = bagOfWords(0,rows)
X = bow['X']
y = bow['y'] 

# Sequential module is used to initialise artificial neural network        
from keras.models import Sequential
# Add different layers to artificial neural network     
from keras.layers import Dense

def create_Model() :
    # Initializing the ANN
    classifier = Sequential()
    
    # Adding input layer and the first hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu',input_dim=1500))
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu'))
    
    # Adding the third hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu'))
    
    # Adding the fourth hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu'))
    
    
    # Adding the Fifth hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu'))
    
    
    # Adding the Sixth hidden layer
    classifier.add(Dense(output_dim=1503,init='normal',activation='relu'))
    
    # Adding the output layer for multiclass change output dim and activation='softmax'
    classifier.add(Dense(output_dim=6,init='normal',activation='sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return classifier

# Create Artificial Neural Network
nn_model = create_Model()

# Fitting ANN to the Training set
history = nn_model.fit(X,y,validation_split=0.25,batch_size=50,epochs=75)

# Print all the keys
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Predict Model on Test Data 500-1000
bow = bagOfWords(rows,rows*2)
X_test = bow['X']
y_test = bow['y'] 
# Predicting the Test set results
y_pred = np.round(nn_model.predict(X_test))
# Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
def hamming_score(y_test, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_test.shape[0]):
        set_true = set( np.where(y_test[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)
# For comparison sake:
import sklearn.metrics
print('Hamming score: {0}'.format(hamming_score(y_test, y_pred))) 
print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)))
print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_test, y_pred)))
# evaluate the model
scores = nn_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (nn_model.metrics_names[1], scores[1]*100))
# Save the result
np.savetxt('output.csv',y_pred,delimiter=',',fmt="%.0f")