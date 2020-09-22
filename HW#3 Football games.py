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
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#LOAD DATA
train_df = pd.read_csv("/Users/furka/Downloads/trainfootball.csv")
test_df=pd.read_csv("/Users/furka/Downloads/testfootball.csv")

#Data types
train_df.info()

#See the distribution of cat data only since there is no useful numeric data
train_df.describe(include=['O'])

#DATA WRANGLE
#We'd want to remove team names since it doesn't have a predictor feature.
#We'd want to remove the dates since it doesn't have a predictor feature because 
#the football games are played only in fall season, only in three months. Considering
#that temperature doesn't change that much. When the team is away, we have to 
#check the each host team's region temperature and that is computationally not 
#convenient. 
train_df=train_df.drop(columns=['Opponent','Date'])
test_df=test_df.drop(columns=['Opponent','Date'])


#Converting categorical values to numeric values
train_df['Label']=train_df['Label'].map({'Win':1,'Lose':0}).astype(int)
train_df['Is_Opponent_in_AP25_Preseason']=train_df['Is_Opponent_in_AP25_Preseason'].map({'In':1,'Out':0}).astype(int)
train_df['Is_Home_or_Away']=train_df['Is_Home_or_Away'].map({'Home':1,'Away':0}).astype(int)
train_df['Media']=train_df['Media'].map({'1-NBC':0,'2-ESPN':1, '3-FOX':2, '4-ABC':3, '5-CBS':4}).astype(int)
test_df['Label']=test_df['Label'].map({'Win':1,'Lose':0}).astype(int)
test_df['Is_Opponent_in_AP25_Preseason']=test_df['Is_Opponent_in_AP25_Preseason'].map({'In':1,'Out':0}).astype(int)
test_df['Is_Home_or_Away']=test_df['Is_Home_or_Away'].map({'Home':1,'Away':0}).astype(int)
test_df['Media']=test_df['Media'].map({'1-NBC':0,'2-ESPN':1, '3-FOX':2, '4-ABC':3, '5-CBS':4}).astype(int)

#MODEL BUILDING
X_train= train_df.drop(columns=['Label'])
Y_train= train_df["Label"]
X_test= test_df.drop(columns=['Label'])
Y_true=test_df['Label']

#Naive Bayes model build
gnb=GaussianNB()
gnb.fit(X_train, Y_train)
Y_predgnb=gnb.predict(X_test)
#accuracy, precision, recall, and F1 score of the test result on our ground truth
target_names = ['Lose','Win']
Classreportgnb=classification_report(Y_true, Y_predgnb, target_names =target_names)
confmatrixgnb=confusion_matrix(Y_true, Y_predgnb)

#kNN model build
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
Y_predkNN = knn.predict(X_test)
#accuracy, precision, recall, and F1 score of the test result on our ground truth
ClassreportkNN=classification_report(Y_true, Y_predkNN, target_names =target_names)
confmatrixkNN=confusion_matrix(Y_true, Y_predkNN)

#PDF of predictions of gnb and kNN
Y_predgnb=pd.Series(Y_predgnb)
pdfgnb=Y_predgnb.plot.kde()













