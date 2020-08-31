#Load the Library
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Load the data
train_df = pd.read_csv("/Users/furka/Downloads/train.csv")
test_df=pd.read_csv("/Users/furka/Downloads/test.csv")

#The Model Building
Y_train=train_df["Survived"]
features=["Pclass","Sex","SibSp","Parch"]
X_train=pd.get_dummies(train_df[features])
X_test=pd.get_dummies(test_df[features])

model=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train,Y_train)
predictions=model.predict(X_test)

#Evaluation
model.score(X_train,Y_train)
score_rounded= round(model.score(X_train, Y_train)*100, 3)
print(score_rounded)





















test_df['Age']=test_df['Age'].fillna(test_df['Age'].dropna().mean())
train_df['Age']=train_df['Age'].fillna(train_df['Age'].dropna().mean())