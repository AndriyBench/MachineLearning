# Import libraries 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
  
  
# Creating dataset 
import csv

fields = []
rows = []
data=[]
data0 = []
data1 = [] 
counter = 0
with open('NormalSample.csv', 'r') as NS:
    NS_reader = csv.reader(NS)
    
    fields = next(NS_reader)
    
    for row in NS_reader:
        rows.append(row)
     
   # print("Total number of rows: %d"%(NS_reader.line_num))
  
    
   # print('Field names are:' + ','.join(field for field in fields))
    
    for row in rows:
        data.append(float(row[2]))
        if(row[1].isdigit() and int(row[1]) == 1):
            data1.append(float(row[2]))
        if(row[1].isdigit() and int(row[1]) == 0):
            data0.append(float(row[2]))
        counter+=1
                
    #print(data1)  
    #print(data0)  
    #print(counter)
 
s = pd.Series(data)
s0 = pd.Series(data0)
s1 = pd.Series(data1)
print(s.describe())
print(s0.describe())
print(s1.describe())

datadict = {'All values': data, 'Group0':data0, 'Group1':data1}
  
fig, ax = plt.subplots()
ax.boxplot(datadict.values())
ax.set_xticklabels(datadict.keys())
ax.set_title('Question 2')
plt.show()


 
