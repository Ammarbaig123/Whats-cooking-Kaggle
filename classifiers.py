# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:20:17 2019

@author: Asad Baig
"""
#from MultinomialNB import MultinomialNB

import json
import io
import unicodedata
import pandas as pd
from sklearn import preprocessing
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#class classifiers():
    
  #def __init__(self):
   #     self.mn = MultinomialNB()
       
#def strip_accents(train):
 # print( ''.join(c for c in unicodedata.normalize('NFD', train)))
      #  return ''.join(c for c in unicodedata.normalize('NFD', train) if unicodedata.category(c) != 'Mn')
    
def readfile():    
   #train_file = io.open('train.json', 'r')       
   #train_dataaa  = json.loads(strip_accents(train_file.read()))
   train_data = pd.read_json('train.json') 
   #test_data = pd.read_json('test.json') 
   train_data.shape
   print("First five elements in our training sample:")
   train_data.head()
   train_data.dtypes
   display(train_data.head())
   display(train_data.shape)
   return train_data
 
def viewdata(train_data):
   print("Number of cuisine categories: {}".format(len(train_data.cuisine.unique())))
   display(train_data.cuisine.unique())
   print('Maximum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().max())
   print('Minimum Number of Ingredients in a Dish: ',train_data['ingredients'].str.len().min())
    


def dataencoding(train_data): 
   
   cv = CountVectorizer()
   train_data['every_ingredients'] = train_data['ingredients'].map(";".join)
   train_data.head()
   X = cv.fit_transform(train_data['every_ingredients'].values)
   display(X.shape)
   display(train_data.head(n=10))
   print(list(cv.vocabulary_.keys())[:5])
   XX=X.toarray()
   print(XX)
   return XX
   
def labelencoding(train_data):
   le = preprocessing.LabelEncoder()
   
   cuisine_cat = le.fit_transform(train_data.cuisine)
   train_data['cuisine_cat'] = cuisine_cat
   cuisine_cat[:2]
   return cuisine_cat
   
def splitting_data(XX, cuisine_cat):
    X_train, X_test, y_train, y_test = train_test_split(XX, cuisine_cat, test_size=0.3,random_state=109)
    return X_train,y_train, X_test, y_test
    
def MultinomialNB(X_train,y_train, X_test, y_test,train_data):
    from sklearn.naive_bayes import MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    print("NAIVE BAYES Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("NAIVE BAYES CONFUSION MATRIX AND CLASSIFICATION REPORT:")
    s='Naive Bayes'
    genconfusionmatrix(X_test, y_test,y_pred,train_data,mnb,s)
    
    
def SVM(X_train,y_train, X_test, y_test,train_data):
   from sklearn.svm import SVC  
   svclassifier = SVC(kernel='linear')  
   svclassifier.fit(X_train, y_train) 
   y_pred = svclassifier.predict(X_test)  
   print("svm Accuracy:",metrics.accuracy_score(y_test, y_pred))
   print("SVM CONFUSION MATRIX AND CLASSIFICATION REPORT:")
   
   genconfusionmatrix(X_test, y_test,y_pred,train_data,svclassifier )    
   
def SGD(X_train,y_train, X_test, y_test,train_data):   
   from sklearn import linear_model
   from sklearn.linear_model import SGDClassifier
   sgd_clf =linear_model.SGDClassifier()
   sgd_clf.fit(X_train, y_train)
   y_pred = sgd_clf.predict(X_test)
   score=sgd_clf.score(X_test, y_test)
   print("SGD accuracy :",score)
   print("SGD CONFUSION MATRIX AND CLASSIFICATION REPORT:")
   s='SGD'
   genconfusionmatrix(X_test, y_test,y_pred,train_data,sgd_clf,s)
   
def DescionTree(X_train,y_train, X_test, y_test,train_data):   
   from sklearn.tree import DecisionTreeClassifier
   tree = DecisionTreeClassifier()
   tree.fit(X_train, y_train)
   y_pred = tree.predict(X_test)
   score=tree.score(X_test, y_test)
   print("dec tree accuracy :",score)
   print("Descion Tree CONFUSION MATRIX AND CLASSIFICATION REPORT:")
   s='Descion Tree'
   genconfusionmatrix(X_test, y_test,y_pred,train_data,tree,s )
   
def RandomForrest(X_train,y_train, X_test, y_test,train_data):    
   from sklearn.ensemble import RandomForestClassifier
   clf=RandomForestClassifier(n_estimators=100)
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   print("RAndom forrest Accuracy:",metrics.accuracy_score(y_test, y_pred))
   print("Random Forrest CLASSIFICATION REPORT:")
   s='Random forrest'
   genconfusionmatrix(X_test, y_test,y_pred,train_data,clf,s )
  
   

def genconfusionmatrix(X_test, y_test,y_pred,train_data,c,s):
   from sklearn.metrics import confusion_matrix
#matplotlib inline
   import matplotlib.pyplot as plt
   plt.style.use('ggplot')
   plt.figure(figsize=(8, 8))
 
   cm = confusion_matrix(y_test, c.predict(X_test))
   cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  # s=''
   plt.imshow(cm_normalized, interpolation='nearest')
   plt.title(s+" confusion matrix")
   plt.colorbar(shrink=0.3)
   cuisines = train_data['cuisine'].value_counts().index
   tick_marks = np.arange(len(cuisines))
   plt.xticks(tick_marks, cuisines, rotation=90)
   plt.yticks(tick_marks, cuisines)
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   print(classification_report(y_test, y_pred, target_names=cuisines))
   return cuisines
   
def main():
    #strip_accents()
    z=readfile()
    a=dataencoding(z)
    viewdata(z)
    b=labelencoding(z)
    c,d,e,f=splitting_data(a,b)
    MultinomialNB(c,d,e,f,z)
   # SVM(c,d,e,f)
    SGD(c,d,e,f,z)
    RandomForrest(c,d,e,f,z)
main()    
          
    
  
      
        