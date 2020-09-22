#Load the Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load Data
train_df = pd.read_csv("/Users/furka/Downloads/train.csv")
test_df=pd.read_csv("/Users/furka/Downloads/test.csv")

#Available Features (Answer to Q1)
train_df[:0]

#Properties (Answer to Q7)
train_df.describe()

#Categorical features (Answer to Q8)
train_df.describe(include=[object])

#Observing Correlations (Answer to Q9)
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')

#Observing Correlations (Answer to Q10)
train_df[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived')

#Visualizing Correlated Data (Answer to Q11)
sns.FacetGrid(train_df, col='Survived',despine=False).map(plt.hist, 'Age', bins=25)

#Visualizing Correlated Data of Numerical and `Categorical features (Answer to Q12)
sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=2, despine=False).map(plt.hist, 'Age', bins=25)

#Visualizing Correlated Data of Numerical and Categorical features (Answer to Q13)
sns.FacetGrid(train_df, row='Embarked', col='Survived').map(sns.barplot, 'Sex', 'Fare', ci=None)

#Data Wrangling (Answer to Q14 and Q15)
train_df=train_df.drop(['Cabin','Ticket'],axis=1)
test_df=test_df.drop(['Cabin','Ticket'], axis=1)

#Converting Strings to Numerical Values (Answer to Q16)
train_df['Sex']=train_df['Sex'].map({'male':0,'female':1}).astype(int)

#Completing Categorical Feature in Embarked (Answer to Q18)
Em=train_df.Embarked.dropna()
MostOccurence=Em.mode()[0]
train_df['Embarked']=train_df['Embarked'].fillna(MostOccurence)

#Completing and Converting a Numeric Feature for Fare (Answer to Q19)
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].dropna().median())

#Converting Fare Features to Ordinal Values (Answer to Q20)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand']).mean().sort_values(by='FareBand')

#Completing Missing or Null Values in Numerical for Age with KNN (Answer to Q17)
train_df.isna().sum() #To see the amount and locations of missing values
test_df['Age']=test_df['Age'].fillna(test_df['Age'].dropna().mean())
train_df['Age']=train_df['Age'].fillna(train_df['Age'].dropna().mean())


































