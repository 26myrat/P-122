import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os, ssl, time
x=np.load('image.npz')['arr_0']
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=9,train_size=7500, test_size=2500)
x_train_scale=x_train/255.0
x_test_scale=x_test/255.0
clf= LogisticRegression(solver="saga", multi_class= "multinomial")
clf.fit(x_train_scale, y_train)
y_pred=clf.predict(x_test_scale)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy) 