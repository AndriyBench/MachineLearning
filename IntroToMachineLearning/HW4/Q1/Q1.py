import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv("Purchase_Likelihood.csv")



#1 A

col_insurance = df['insurance'].value_counts()
print(col_insurance)
print()
print("-------------------")
print()


#1 B

print(pd.crosstab(df.group_size, df.insurance))
print()
print("-------------------")
print()


#1 C

print(pd.crosstab(df.homeowner, df.insurance))
print()
print("-------------------")
print()


#1 D

print(pd.crosstab(df.married_couple, df.insurance))
print()
print("-------------------")
print()


#1 E

#create table
data = np.array([[115460, 329552, 74293], [25728, 91065, 19600], [2282, 5069, 1505], [221, 381, 93]])
data1 = np.array([[78659, 183130, 46734], [65032, 242937, 48757]])
data2 = np.array([[117110, 333272, 75310], [26581, 92795, 20181]])


#Chi-squared test statistic, sample size, and minimum of rows and columns
X1 = stats.chi2_contingency(data, correction=False)[0]
n = np.sum(data)
minDim = min(data.shape)-1


X2 = stats.chi2_contingency(data1, correction=False)[0]
n = np.sum(data1)
minDim = min(data1.shape)-1


X3 = stats.chi2_contingency(data2, correction=False)[0]
n = np.sum(data2)
minDim = min(data2.shape)-1

#calculate Cramer's V 
V1 = np.sqrt((X1/n) / minDim)
V2 = np.sqrt((X2/n) / minDim)
V3 = np.sqrt((X3/n) / minDim)

#display Cramer's V
print("group_size " + str(V1))
print("homeowner " + str(V2))
print("married_couple " + str(V3))


#1 F



