
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sklearn


data=pd.read_csv('C:/Users/govar/OneDrive/Desktop/files/demo/progam.py/train.csv')
data.isnull()


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data)


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=data,palette='RdBu_r')


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=data,palette='rainbow')


sns.distplot(data['Age'].dropna(),kde=False,color='blue',bins=40)

sns.distplot(data['Age'].dropna(), kde=False, color='blue', bins=40)


data['Fare'].hist(color='green', bins=40, figsize=(8, 4))


data = pd.read_csv(
    'C:/Users/govar/OneDrive/Desktop/files/demo/progam.py/train.csv')

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data = pd.get_dummies(
    data, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

features = ['Age', 'Sex_male', 'Pclass_2', 'Pclass_3']
X = data[features]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")


coefficients = model.coef_[0]
feature_importance = pd.DataFrame(
    {'Feature': features, 'Coefficient': coefficients})
print("Feature Importance:")
print(feature_importance)
