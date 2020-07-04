#===========================IMPORTING LIBRARIES AND PACKAGES, SOURCE FILES============================
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#=========================Importing the dataframe and reading it into the project=========================
my_data = pd.read_csv("./drug200.csv", delimiter=",")
my_data[0:5]
df = my_data.head()
print(df)

#======================================Getting the data size===================================
dsize = df.shape
print(dsize)

#==========================