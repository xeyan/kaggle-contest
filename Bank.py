import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix

warnings.filterwarnings('ignore')
df = pd.read_csv('D:/python/kaggle dataset/bank/dataset.csv')

label_encoder = LabelEncoder()

num_cols = df.select_dtypes(include=['int64']).columns
cat_cols = df.select_dtypes(exclude=['int64']).columns
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])
X = df.drop(columns=['conversion_status'])
y = df['conversion_status']
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)

RF = RandomForestClassifier(random_state=42)
RF.fit(X_train,Y_train)

RandomForestClassifier(random_state=42)
pred_RF = RF.predict(X_test)
confusion_matrix(Y_test,pred_RF)

print(f'accuracy : {accuracy_score(Y_test,pred_RF)}')

LGR = LogisticRegression()
LGR.fit(X_train, Y_train)

LogisticRegression()
pred_LGR = LGR.predict(X_test)
confusion_matrix(Y_test,pred_LGR)

print(f'accuracy : {accuracy_score(Y_test,pred_LGR)}')
