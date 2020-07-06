#===========================IMPORTING LIBRARIES AND PACKAGES, SOURCE FILES============================
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

#=========================Importing the dataframe and reading it into the project=========================
my_data = pd.read_csv("./drug200.csv", delimiter=",")
my_data[0:5]
df = my_data.head()
print(df)

#======================================Getting the data size===================================
dsize = df.shape
print(dsize)



#======================================DATA PRE-PROCESSING========================================

#====================================Viewing the data===========================
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
pre = X[0:5]
print(pre)

#============================Convert categorical variables to numerical values=============================
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]
preprocessed = X[0:5]
print(preprocessed)

#==================================Adding the target variables======================================
y = my_data["Drug"]
y[0:5]
target = y[0:5]
print(target)



#==================================SETTING UP THE DECISION TREE====================================