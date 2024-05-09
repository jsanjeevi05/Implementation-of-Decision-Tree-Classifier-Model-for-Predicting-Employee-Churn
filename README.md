# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANJEEVI J
RegisterNumber: 212222110040
*/
```
```python
import pandas as pd
df=pd.read_csv("CSVs/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
      'time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=df['left']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
accuracy=metrics.accuracy_score(Ytest,Ypred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![decision tree classifier model](sam.png)
![Screenshot 2024-04-25 094224](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/40000f88-0c18-4bec-ab54-bb01ccb86807)
![Screenshot 2024-04-25 094246](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/a9b2d4da-d4e9-4a88-9042-518aef46cfbc)
![Screenshot 2024-04-25 094253](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/f6b41b1d-e51f-4f92-8543-2982ee12450d)
![Screenshot 2024-04-25 094258](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/ad9224c6-a0a9-48c2-9105-148b1c235f4a)
![Screenshot 2024-04-25 094326](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/aae6e273-959d-4a70-8d17-d0523b2b1453)
![Screenshot 2024-04-25 094338](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/32b58c60-b80c-43b9-aab4-48c01ef3e5b5)
![Screenshot 2024-04-25 094345](https://github.com/Darshans05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115534676/29fa006b-47ae-45f4-ae83-489e97977d40)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
