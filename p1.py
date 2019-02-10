#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 15:56:49 2019

@author: kaiwei
"""

"""
Tube Spam classification
@author: kaiwei
# -*- coding: utf-8 -*-

Data description:
CLASS: 0 is ham(legitimate comment), 1 is spam

"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import *


## loop throught data on github
df = pd.DataFrame()
url_list = ['Youtube01-Psy.csv', 'Youtube02-KatyPerry.csv', 'Youtube03-LMFAO.csv','Youtube04-Eminem.csv', 'Youtube05-Shakira.csv']
for url_i in url_list:
    url = 'https://raw.githubusercontent.com/Kai-Wei-626/ML/master/' + url_i
    df_i = pd.read_csv(url)
    df = pd.concat([df, df_i])

df = df.reset_index()
df = df.drop(['index'], axis = 1)




## convert text to bags of word representation
text = df.CONTENT
count = CountVectorizer()
bag_of_words = count.fit_transform(text)

# Show feature matrix
bag_of_words.toarray()

# Get feature names
feature_names = count.get_feature_names()

# Create data frame
df_text = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

# join the text back
df_total = pd.concat([df, df_text], axis = 1) 

# start training
y = df.CLASS
X = df_text

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## decision tree
def DTClassfier(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    
    print('Decision Tree: ')
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train, y_pred)*100)
    print('precision: ', precision_score(y_train, y_pred)*100)
    print('recall: ', recall_score(y_train, y_pred)*100)
    print('F1_score: ', f1_score(y_train, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    

DTClassfier(X_train, X_test, y_train, y_test)

# NN
def NNClassifier(X_train, X_test, y_train, y_test):    
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    
    print('Neural Net: ')
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train, y_pred)*100)
    print('precision: ', precision_score(y_train, y_pred)*100)
    print('recall: ', recall_score(y_train, y_pred)*100)
    print('F1_score: ', f1_score(y_train, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    

    
NNClassifier(X_train, X_test, y_train, y_test)




def Boosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(max_depth=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train, y_pred)*100)
    print('precision: ', precision_score(y_train, y_pred)*100)
    print('recall: ', recall_score(y_train, y_pred)*100)
    print('F1_score: ', f1_score(y_train, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    
    
Boosting(X_train, X_test, y_train, y_test)





'''
running a PCA to reduce the dimension to 2 so that I can plot the PC to the scatter chart
and determine which SVM kernal I can apply
'''

X_pca = X.reset_index().drop(['level_0'], axis = 1)
X_pca.head()
y_pca = y.reset_index().drop(['index'], axis = 1)
y_pca.head()


def PCA_2(X, y):
    '''
    input: X, y
    output: a df has 2 principal components
    '''
    
    pca = PCA(n_components=2)
    
    principalComponents = pca.fit_transform(X)
    
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])
    
    finalDf = pd.concat([principalDf,y], axis = 1)
    
    return finalDf



finalDf = PCA_2(X_pca, y_pca)
final_X = finalDf[['principal component 1', 'principal component 2']]
final_y = finalDf['CLASS']

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(final_X, final_y, test_size=0.2, random_state=42)


# PCA scatter plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['CLASS'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# SVM training start here
def SVM_Kernal(X_train, X_test, y_train, y_test, kernel = 'rbf'):
    clf = SVC(gamma='auto', kernel= kernel)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_train)
    print('SVM: ')
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train, y_pred)*100)
    print('precision: ', precision_score(y_train, y_pred)*100)
    print('recall: ', recall_score(y_train, y_pred)*100)
    print('F1_score: ', f1_score(y_train, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    
    
SVM_Kernal(X_train, X_test, y_train, y_test, kernel = 'linear')

'''
training data metrics: 
accuracy score:  99.87212276214834
precision:  100.0
recall:  99.74651457541192
testing data metrics: 
accuracy score:  94.89795918367348
precision:  97.11538461538461
recall:  93.51851851851852
'''



def KNN(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_train)
    print('KNN: ')
    print('training data metrics: ')
    print('accuracy score: ', accuracy_score(y_train, y_pred)*100)
    print('precision: ', precision_score(y_train, y_pred)*100)
    print('recall: ', recall_score(y_train, y_pred)*100)
    print('F1_score: ', f1_score(y_train, y_pred)*100)    
    #print('confusion matrix: ',   confusion_matrix(y_train, y_pred)) 
    y_pred_test = clf.predict(X_test)
    print('testing data metrics: ')
    print('accuracy score: ', accuracy_score(y_test, y_pred_test)*100)
    print('precision: ', precision_score(y_test, y_pred_test)*100)
    print('recall: ', recall_score(y_test, y_pred_test)*100)
    print('F1_score: ', f1_score(y_test, y_pred_test)*100)    




