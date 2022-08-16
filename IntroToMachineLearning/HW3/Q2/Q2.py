import numpy as np
import pandas as pd
import math

from sklearn import metrics
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#load data

df = pd.read_csv("claim_history.csv")

features = df[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]

#print(features.head())

#Split for training and testing data
features_train,features_test,labels_train, labels_test = train_test_split(features,df["CAR_USE"],test_size = 0.3, random_state=27513,stratify = df["CAR_USE"])


cross_Table_Train = pd.crosstab(labels_train,columns =  ["Count"],margins=True,dropna=True)
cross_Table_Train["Proportions"] = (cross_Table_Train["Count"]/len(labels_train))*100
print(cross_Table_Train)

cross_Table_test = pd.crosstab(labels_test,columns =  ["Count"],margins=True,dropna=True)
cross_Table_test["Proportions"] = (cross_Table_test["Count"]/len(labels_test))*100
print(cross_Table_test)


#what is the P(Train | Car Use = Commercial)

c=0
prob_train = len(features_train)/len(df["CAR_USE"])

for i in df["CAR_USE"]:
    if(i == "Commercial"):
        c+=1
probC = (prob_train*c/len(df["CAR_USE"]))/(c/len(df["CAR_USE"]))

#print("The probability that an observation is in the Training partition given that CAR_USE = Commercial is", probC)

#what is the P(Train | Car Use = Private)

p=0
prob_test = len(features_test)/len(df["CAR_USE"])

for i in df["CAR_USE"]:
    if(i == "Private"):
        p+=1
probP = (prob_test*c/len(df["CAR_USE"]))/(c/len(df["CAR_USE"]))

#print("The probability that an observation is in the Training partition given that CAR_USE = Commercial is", probP)

features_train["Labels"] = labels_train

#calculating the entropy of the root node

ec = 0
for i in df["CAR_USE"]:
    if(i == "Commercial"):
        ec+=1
probC =  ec / len(df["CAR_USE"])
probP = (len(df["CAR_USE"]) - ec)/len(df["CAR_USE"])
ans = -((probC * np.log2(probC) + probP * np.log2(probP)))
print("Entropy for root node is given as",ans)



# all possible combuinations for occupation

occ_col = df["OCCUPATION"].unique()
occ_comb = []
for i in range(1, math.ceil(len(occ_col)/2)):
    occ_comb +=list(combinations(occ_col, i))
    
#print(occ_comb)

# all possible cimbinations for car type
ctype_col = df["CAR_TYPE"].unique()
ctype_comb = []

for i in range(1, math.ceil(len(ctype_col)/2)+1):
    x = list(combinations(ctype_col,i))
    if i == 3:
        x = x[:10]
    ctype_comb.extend(x) 

#All possible combinations for education
ed_comb = [("Below High School",),("Below High School","High School",),("Below High School","High School","Bachelors",),("Below High School","High School","Bachelors","Masters",)]
    
def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

   #print(split)
   dataTable = inData
   dataTable['LE_Split'] = False
   for k in dataTable.index:
       if dataTable.iloc[:,0][k] in split:
           dataTable['LE_Split'][k] = True
   #print(dataTable['LE_Split'])
   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   #print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableEntropy = 0
   for iRow in range(nRows-1):
      rowEntropy = 0
      for iColumn in range(nColumns):
         proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
         if (proportion > 0):
            rowEntropy -= proportion * np.log2(proportion)
      #print('Row = ', iRow, 'Entropy =', rowEntropy)
      #print(' ')
      tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
   tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
   return(tableEntropy)


def calculate_min_entropy(df,variable,combinations):
    inData1 = df[[variable,"Labels"]]
    entropies = []
    for i in combinations:
        EV = EntropyIntervalSplit(inData1, list(i))
        entropies.append((EV,i))
    return min(entropies)
print("")
print("-------------------------------------------")
print("Entropy calculations for split - first level")
print("-------------------------------------------")

 
    
#Entropy for best split in Occupation
entropy_occupation = calculate_min_entropy(features_train,"OCCUPATION",occ_comb)
print(entropy_occupation)

#Entropy for best split in Car Type
entropy_cartype = calculate_min_entropy(features_train,"CAR_TYPE",ctype_comb)
print(entropy_cartype)

#Entropy for best split in EDUCATION
entropy_education = calculate_min_entropy(features_train,"EDUCATION",ed_comb)
print(entropy_education)
#print(df.head())

#Split the dataframes according to the given split. If the observation belongs 
#to ('Blue Collar', 'Unknown', 'Student'), then it will be in the left dataframe, 
#else it will be in the right dataframe. 
#In this way, we create a parent node where we ask question about the occupation and split our data into two child nodes.
#Here, we are done with level 1.

df_1_left = features_train[(features_train["OCCUPATION"] == "Blue Collar") | (features_train["OCCUPATION"] == "Unknown") | (features_train["OCCUPATION"] == "Student")]
df_1_right =  features_train[(features_train["OCCUPATION"] != "Blue Collar") & (features_train["OCCUPATION"] != "Unknown") & (features_train["OCCUPATION"] != "Student")]

df_1_left_ct = features_train[(features_train["CAR_TYPE"] == "Minivan") | (features_train["CAR_TYPE"] == "SUV") | (features_train["CAR_TYPE"] == "Sports Car")]
df_1_right_ct =  features_train[(features_train["CAR_TYPE"] != "Minivan") & (features_train["CAR_TYPE"] != "SUV") & (features_train["CAR_TYPE"] != "Sports Car")]

df_1_left_e = features_train[(features_train["EDUCATION"] == "Below High School")]
df_1_right_e =  features_train[(features_train["EDUCATION"] != "Below High School")]

print("")
print("-------------------------------------------")
print("Values in branches - first level")
print("-------------------------------------------")

print(len(df_1_right),len(df_1_left))
print(len(df_1_right_ct),len(df_1_left_ct))
print(len(df_1_right_e),len(df_1_left_e))



#level 2 ???

#entropy calculations for left split

print("-------------------------------------------")
print("Entropy calculations for left split - second level")
print("-------------------------------------------")

left_edu_entropy = calculate_min_entropy(df_1_left,"EDUCATION",ed_comb)
print(left_edu_entropy)

left_ct_entropy = calculate_min_entropy(df_1_left,"CAR_TYPE",ctype_comb)
print(left_ct_entropy)

occupation_column = ['Blue Collar', 'Unknown', 'Student']
occ_comb = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occ_comb+=list(combinations(occupation_column,i))
left_occupation_entropy = calculate_min_entropy(df_1_left,"OCCUPATION",occ_comb)
print(occ_comb)





#Entropy calculations for right split

occupation_column = ['Professional', 'Manager', 'Clerical', 'Doctor','Lawyer','Home Maker']
occ_comb = []
for i in range(1,math.ceil(len(occupation_column)/2)):
    occ_comb+=list(combinations(occupation_column,i))
right_occupation_entropy = calculate_min_entropy(df_1_right,"OCCUPATION",occ_comb)

right_edu_entropy = calculate_min_entropy(df_1_right,"EDUCATION",ed_comb)
right_ct_entropy = calculate_min_entropy(df_1_right,"CAR_TYPE",ctype_comb)

print("-------------------------------------------")
print("Entropy calculations for right split - second level")
print("-------------------------------------------")
print(right_ct_entropy)
print(right_edu_entropy)
print(right_occupation_entropy)



df_2_left_left = df_1_left[(features_train["EDUCATION"] == "Below High School")]
df_2_left_right = df_1_left[(features_train["EDUCATION"] != "Below High School")]


cnt = 0
for i in df_2_left_left["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial = cnt/len(df_2_left_left["Labels"])
print("Count of commercial and private is",cnt,(len(df_2_left_left)-cnt),"respectively and probability of the event",proba_commercial)

cnt = 0
for i in df_2_left_right["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial = cnt/len(df_2_left_right["Labels"])
print("Count of commercial and private is",cnt,(len(df_2_left_right)-cnt),"respectively and probability of the event",proba_commercial)

df_2_right_left = df_1_right[(features_train["CAR_TYPE"] == "Minivan") | (features_train["CAR_TYPE"] == "Sports Car") | (features_train["CAR_TYPE"] == "SUV")]
df_2_right_right = df_1_right[(features_train["CAR_TYPE"] != "Minivan") & (features_train["CAR_TYPE"] != "Sports Car") & (features_train["CAR_TYPE"] != "SUV")]

cnt = 0
for i in df_2_right_left["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial = cnt/len(df_2_right_left["Labels"])
1-proba_commercial
print("Count of commercial and private is",cnt,(len(df_2_right_left)-cnt),"respectively and probability of the event",proba_commercial)

cnt = 0
for i in df_2_right_right["Labels"]:
    if i == "Commercial":
        cnt+=1
proba_commercial = cnt/len(df_2_right_right["Labels"])
proba_commercial
print("Count of commercial and private is",cnt,(len(df_2_right_right)-cnt),"respectively and probability of the event",proba_commercial)

#Thresold probability of the event from training set
cnt = 0
for i in features_train["Labels"]:
    if i == "Commercial":
        cnt+=1
threshold = cnt/len(features_train["Labels"])
print("Threshold probability of an event is given as",threshold)


predicted_probability=[]
occ = ["Blue Collar","Student","Unknown"]
edu = ["Below High School",]
cartype = ["Minivan","SUV","Sports Car"]
for k in features_test.index:
    if features_test.iloc[:,1][k] in occ:
            if features_test.iloc[:,2][k] in edu:
                predicted_probability.append(0.24647887323943662)  #Leftmost Leaf Node
            else:
                predicted_probability.append(0.8504761904761905)   #Right leaf from left subtree
    else:
            if features_test.iloc[:,0][k] in cartype:
                predicted_probability.append(0.006151953245155337)  #Left leaf from right subtree
            else:
                predicted_probability.append(0.5464396284829721)   #Rightmost Leaf Node
           
#print(predicted_probability)
                
prediction = []
for i in range(0,len(labels_test)):
    if predicted_probability[i] >= threshold :
        prediction.append("Commercial")
    else:
        prediction.append("Private")
 
print("Predicted probability for CAR_USE: ", accuracy_score(labels_test, prediction))
#print(prediction)