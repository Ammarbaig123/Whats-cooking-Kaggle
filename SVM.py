
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

def fileReader():
    train = json.load(open('train.json'))
    test = json.load(open('test.json'))

    train_as_text = [' '.join(sample['ingredients']).lower() for sample in train]
    train_cuisine = [sample['cuisine'] for sample in train]

    test_as_text = [' '.join(sample['ingredients']).lower() for sample in test]

    return train,test,train_as_text,train_cuisine,test_as_text
def SVM(train,test,train_as_text,train_cuisine,test_as_text):
    tfidf_enc = TfidfVectorizer(binary=True)
    lbl_enc = LabelEncoder()

    X = tfidf_enc.fit_transform(train_as_text)
    X = X.astype('float16')

    X_test = tfidf_enc.transform(test_as_text)
    X_test = X_test.astype('float16')

    y = lbl_enc.fit_transform(train_cuisine)

    clf = SVC(gamma='auto', C=1000)

    model = OneVsRestClassifier(clf, n_jobs=4)
    model.fit(X, y)
    y_test = model.predict(X_test)

    test_cuisine = lbl_enc.inverse_transform(y_test)
    test_id = [sample['id'] for sample in test]
   # print(test_id)
    return test_id,test_cuisine,model,X,y
#03332218660

def submission(test_id,test_cuisine):
    submission_df = pd.DataFrame({'id': test_id, 'cuisine': test_cuisine}, columns=['id', 'cuisine'])
    submission_df.to_csv('svm.csv', index=False)

def accuracy(model,X,y):
    print('Accuracy on train data = ', format(model.score(X, y)))


def SVMHandler():
    train,test,train_as_text,train_cuisine,test_as_text = fileReader()
    test_id,test_cuisine,model,X,y = SVM(train,test,train_as_text,train_cuisine,test_as_text)
    submission(test_id, test_cuisine)
    accuracy(model, X, y)

SVMHandler()