
import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt

fields = []
rows = []

data = []

temp1 = 0.0
temp2 = 0.0
variance = 0.0
with open('NormalSample.csv', 'r') as NS:
    NS_reader = csv.reader(NS)
    
    fields = next(NS_reader)
    
    for row in NS_reader:
        rows.append(row)
     
   # print("Total number of rows: %d"%(NS_reader.line_num))
  
    
   # print('Field names are:' + ','.join(field for field in fields))
    
    
    for row in rows:
        for col in row:
            if(col.isdigit() == False):
                data.append(float(col))
                
# Generate a frequence table
uvalue, ucount = numpy.unique(data, return_counts = True)
print('Unique Values:\n', uvalue)
print('Unique Counts:\n', ucount)


# Draw a better labeled histogram with specified bin boundaries
plt.hist(data, bins = [26,26.5,27,27.5,28,28.5,29,29.5,30,30.5,31,31.5,32,32.5,33,33.5,34,34.5,35,35.5], align='mid')
plt.title('x histogram')
plt.xlabel('Number of bins')
plt.ylabel('x value')
plt.grid(axis = 'y')
plt.show()

#print(data)
df = pd.read_csv('NormalSample.csv')
df = df[['i', 'group', 'x']]



#print(df.head(10))
print(df.describe(include='all'))
