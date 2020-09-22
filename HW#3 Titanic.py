#Load the Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import metrics

#Load Data
train_df = pd.read_csv("/Users/furka/Downloads/train.csv")
test_df=pd.read_csv("/Users/furka/Downloads/test.csv")

#Categorical features
train_df.describe(include=[object])

#Observing Correlations
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')
train_df[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived')
train_df[['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived')
train_df[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived')

#Data Wrangling
train_df=train_df.drop(['Cabin','Ticket','SibSp','Parch','Name','PassengerId'],axis=1)
test_df=test_df.drop(['Cabin','Ticket','SibSp','Parch','Name','PassengerId'], axis=1)

#Completing Categorical Feature in Embarked
Em=train_df.Embarked.dropna()
MostOccurence=Em.mode()[0]
train_df['Embarked']=train_df['Embarked'].fillna(MostOccurence)

#Converting Strings to Numerical Value
train_df['Sex']=train_df['Sex'].map({'male':0,'female':1}).astype(int)
test_df['Sex']=test_df['Sex'].map({'male':0,'female':1}).astype(int)
train_df['Embarked']=train_df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
test_df['Embarked']=test_df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

#Completing Fare
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].dropna().median())
train_df['Fare']=train_df['Fare'].fillna(train_df['Fare'].dropna().median())

#Completing Missing or Null Values in Numerical for Age with median 
train_df.isna().sum() #To see the amount and locations of missing values
test_df['Age']=test_df['Age'].fillna(test_df['Age'].dropna().mean())
train_df['Age']=train_df['Age'].fillna(train_df['Age'].dropna().mean())

X_train= train_df.drop(columns=['Survived'])
Y_train= train_df["Survived"]
X_test= test_df

#Naive Bayes model build
gnb=GaussianNB()
gnb.fit(X_train, Y_train)
Y_predgnb=gnb.predict(X_test)

#average Accuracy, Precision, Recall, and F1 score of the five-fold 
#cross validation
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}

results = cross_validate(estimator=gnb,
                                          X=X_test,
                                          y=Y_predgnb,
                                          cv=5,
                                          scoring=scoring)

np.mean(results['test_accuracy'])
np.mean(results['test_precision'])
np.mean(results['test_recall'])
np.mean(results['test_f1_score'])

#Plot of kNN
# try K=1 through K=70 and record testing accuracy
k_range = range(1,70)
k_scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_predkNN = knn.predict(X_test)
    accuracy_kNN = cross_val_score(knn, X_test, Y_predkNN, cv = 5, scoring='accuracy')
    k_scores.append(accuracy_kNN.mean())

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(x_axis, y_axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')


















