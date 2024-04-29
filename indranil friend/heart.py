import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns

#reading the heartset
heart = pd.read_csv(r'C:\Users\91743\.dotnet\Desktop\heart.csv')

#getting info of the heartset 
print(heart.info())
print(heart.isnull().sum())

#checking for duplicate value
heart_dup = heart.duplicated().any()
print(heart_dup)

heart = heart.drop_duplicates()
print(heart)

print(heart.describe())

#checking correlation between columns
print(heart.corr()['target'].abs().sort_values(ascending=False))
#checking distribution of target values 
print(heart['target'].value_counts())
sns.set_style("whitegrid")

plt.figure(figsize=(8,6))
sns.countplot(x='target', data=heart, palette='pastel')
plt.title('Target Value Distribution')
plt.show()

# DAta analysis between sex and target i.e chance of heart disease
print(heart['sex'].unique())

sns.set_style("whitegrid")

plt.figure(figsize=(8,6))
sns.barplot(x='sex', y='target',data=heart, palette='pastel')
plt.title('Relation b/w Sex and target')
plt.show()

# Data analysis between chest pain(cp) and target i.e chance of heart disease
print(heart['cp'].unique())

sns.set_style('whitegrid')

plt.figure(figsize=(8,6))
sns.barplot(x='cp', y='target', data=heart, palette='pastel')
plt.title('Relation b/w Chest Pain and Target')
plt.show()
#get correlations of each features in heartset
corrmat = heart.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(50,50))

#plot heat map of Overall Dataset
g=sns.heatmap(heart[top_corr_features].corr(),cmap="RdYlGn",annot=True)
plt.show()

