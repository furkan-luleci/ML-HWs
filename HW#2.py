#Load the Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree


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

#Visualizing Correlated Data
sns.FacetGrid(train_df, col='Survived',despine=False).map(plt.hist, 'Age', bins=25)

#Visualizing Correlated Data of Numerical and `Categorical features
sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=2, despine=False).map(plt.hist, 'Age', bins=25)

#Visualizing Correlated Data of Numerical and Categorical feature
sns.FacetGrid(train_df, row='Embarked', col='Survived').map(sns.barplot, 'Sex', 'Fare', ci=None)

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


#ML model building - Decision Tree
X_train= train_df.drop(columns=['Survived'])
Y_train= train_df["Survived"]
X_test= test_df
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, Y_train)
Y_prediction_DT= decisiontree.predict(X_test)

#Decision Tree Plotting
plt.figure(figsize=(120,80))
a = plot_tree(decisiontree, filled=True, rounded=True, fontsize=24)

#ML model building - Random Forest
randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(X_train, Y_train)
Y_prediction_RF= randomforest.predict(X_test)

#Accuracy five-fold cross validation for Decision Tree
accuracy_DT = cross_val_score(decisiontree, X_test, Y_prediction_DT, scoring='accuracy', cv = 5)
print("Accuracy of Model with Cross Validation - DT is:",accuracy_DT.mean() * 100)

#Accuracy five-fold cross validation for Random Forest
accuracy_RF = cross_val_score(decisiontree, X_test, Y_prediction_RF, scoring='accuracy', cv = 5)
print("Accuracy of Model with Cross Validation - RF is:",accuracy_RF.mean() * 100)


























