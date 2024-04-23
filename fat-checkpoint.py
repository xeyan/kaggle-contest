import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sb

import tensorflow as tf
from tensorflow.python import keras

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

df = pd.read_csv('D:/python/kaggle dataset/fat/eating.csv')
df["NObeyesdad"] = LabelEncoder().fit_transform(df["NObeyesdad"])
X = df.drop("NObeyesdad", axis = 1)
y = tf.keras.utils.to_categorical(df["NObeyesdad"])
X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns = X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 42)

modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(24, input_shape = (X_train.shape[1], ), activation = "relu"))
modelo.add(tf.keras.layers.Dense(12, activation = "relu"))
modelo.add(tf.keras.layers.Dense(7, activation = "softmax"))
print(modelo.summary())
modelo.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)
modelo.fit(X_train, y_train, validation_split = 0.2, batch_size = 100, epochs = 600, callbacks = [early_stop])
df_historia = pd.DataFrame(modelo.history.history)
y_pred = modelo.predict(X_test)
print("La calidad del modelo es de: ", accuracy_score(y_test, y_pred) * 100, "%")

