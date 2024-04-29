import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns

#reading the dataset
heart = pd.read_csv(r'C:\Users\91743\.dotnet\Desktop\heart.csv')

#getting info of the dataset 
print(heart.info())
print(heart.isnull().sum())

#checking distribution of target values 
print(heart['target'].value_counts())

#get correlations of each features in dataset
corrmat = heart.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(50,50))

#plot heat map
g=sns.heatmap(heart[top_corr_features].corr(),cmap="RdYlGn",annot=True)
plt.show()

