import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('https://raw.githubusercontent.com/Kai-Wei-626/ML/master/titanic/titanic_train.csv')


# ## EDA
print('Overall Survival Rate:',df['Survived'].mean())
print('Missing values:',df.isnull().sum())
print('\n')

df_missing_val = df[df['Age'].isnull()]
df_without_missing_val = df[df['Age'].isnull() == False]
print('Passengers with missing age Survival Rate: ', df_missing_val.Survived.mean())
print('Passengers without missing age Survival Rate: ', df_without_missing_val.Survived.mean())

print('\n')
df_missing_val = df[df['Cabin'].isnull()]
df_without_missing_val = df[df['Cabin'].isnull() == False]
print('Passengers with missing Cabin Survival Rate: ', df_missing_val.Survived.mean())
print('Passengers without missing Cabin Survival Rate: ', df_without_missing_val.Survived.mean())

print('\n')
print('Conclusion: Passengers with missing value seems having low survival rate')

# get_dummies function
def dummies(col,df):
    df_dum = pd.get_dummies(df[col])
    #test_dum = pd.get_dummies(test[col])
    df = pd.concat([df, df_dum], axis=1)
    #test = pd.concat([test,test_dum],axis=1)
    df.drop(col,axis=1,inplace=True)
    #test.drop(col,axis=1,inplace=True)
    return df

# text object without much value, dropped
dropping = ['PassengerId', 'Name', 'Ticket']
df.drop(dropping,axis=1, inplace=True) # inplace=True make sure it's not just printing, it's in place dropping

# age 
# for missing val, create flag for them
def age_missing(row):
    result = 0.0
    if np.isnan(row):
        result = 1.0
    return result

df['age_missing'] = df['Age'].apply(age_missing)

# Also imputing missing value with random int
nan_num = df['Age'].isnull().sum()
age_mean = df['Age'].mean()
age_std = df['Age'].std()
filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)
df['Age'][df['Age'].isnull()==True] = filling
nan_num = df['Age'].isnull().sum()


# Cabin
# for missing value, we create a lable called 'm' for them. 
# Plus, for other rows, extract first character as the label for that cabin.
def Cabin_missing(row):
    result = 'Cabin_' + row[:1]
    if row == 'nan':
        result = 'Cabin_M'
    return result

df.Cabin = df.Cabin.apply(str)

df.Cabin = df.Cabin.apply(Cabin_missing)

df = dummies('Cabin',df)


#Embark
# fill the majority val,'s', into missing val col
df['Embarked'].fillna('S',inplace=True)

def Embarked_add_name(row):
    if row == 'S':
        result = 'Embarked_S'
    elif row == 'C':
        result = 'Embarked_C'
    elif row == 'Q':
        result = 'Embarked_Q'
    return result

df.Embarked = df.Embarked.apply(Embarked_add_name)

# one hot encoded Embarked
df = dummies('Embarked',df)


# one hot encoded Sex
df = dummies('Sex', df)



# create feature family size
df['Family_Size']=df['SibSp']+df['Parch']
# create feature Fare per person
df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
# This is an interaction term, since age and class are both numbers we can just multiply them.
df['Age*Class']=df['Age']*df['Pclass']


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

train = df.sample(frac=0.8, random_state = 1)
test = df.loc[~df.index.isin(train.index)]




# ## Modeling

# import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV



train_X=train.drop('Survived',axis=1)
train_y=train['Survived']
test_X=test.drop('Survived',axis=1)
test_y=test['Survived']


print(train_X.shape, train_y.shape)
print(test_X.shape,test_y.shape)



## Decision tree
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

DTClassfier(train_X, test_X, train_y, test_y)

from sklearn import tree
import graphviz 
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state=42)
clf.fit(train_X, train_y)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render('titanic') 

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=train_X.columns,  
                     class_names=train_y.name,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data) 




def NNClassifier(X_train, X_test, y_train, y_test):    
    clf = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(11,), random_state=1,max_iter = 1500)
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

    
NNClassifier(train_X, test_X, train_y, test_y)


# In[321]:


'''
parameters = {'solver': ['lbfgs'], 'max_iter': [500,1000,1500], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12)}
clf_grid = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1) # n-jobs=-1 means using all processors
clf_grid.fit(train_X,train_y)

print("-----------------Original Features--------------------")
print("Best score: %0.4f" % clf_grid.best_score_)
print("Using the following parameters:")
print(clf_grid.best_params_)


output:
Best score: 0.8149
Using the following parameters:
{'alpha': 0.001, 'hidden_layer_sizes': 11, 'max_iter': 1500, 'solver': 'lbfgs'}
'''

#NN accuracy based on iteration
scores_train = []
scores_test = []
for iterations in range(100, 1500, 100):
    mlp = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(11,), random_state=1,max_iter = iterations)
    mlp.fit(train_X, train_y)
    y_pred = mlp.predict(train_X)
    y_pred_test = mlp.predict(test_X)
    
    scores_train.append(accuracy_score(train_y, y_pred)*100)
    scores_test.append(accuracy_score(test_y, y_pred_test)*100)

""" Plot """
fig, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(scores_train)
ax[0].set_title('Train')

ax[1].plot(scores_test)
ax[1].set_title('Test')
ax[1].set_xticks(range(len(list(range(100, 1500, 100)))))
ax[1].set_xticklabels(list(range(100, 1500, 100)), rotation = 90)

fig.suptitle("Accuracy over iterations", fontsize=14)



#boosting
def Boosting(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(loss='exponential',learning_rate =0.002, max_depth=2, n_estimators = 10000,  n_iter_no_change = 10000, random_state =42)
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
    
Boosting(train_X, test_X, train_y, test_y)


# In[ ]:


# SVM training start here
def SVM_Kernal(X_train, X_test, y_train, y_test, kernel = 'rbf'):
    clf = SVC(kernel= kernel)
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
    
SVM_Kernal(train_X, test_X, train_y, test_y, kernel = 'rbf')


# In[310]:


## K NN
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

KNN(train_X, test_X, train_y, test_y)

