import pandas as pd
import numpy
from sklearn.neighbors import NearestNeighbors as kNN


fields = []
rows = []


import csv

fields = []
rows = []
data = []
i=0
x =0

with open('Fraud.csv', 'r') as NS:
    NS_reader = csv.reader(NS)
    
    fields = next(NS_reader)
    
    for row in NS_reader:
        rows.append(row)
        
   # print("Total number of rows: %d"%(NS_reader.line_num))
  
    
   # print('Field names are:' + ','.join(field for field in fields))
    
    
   # for row in rows:
    #    data.append(int(row[1]))
    ##    if(data[i] == 1):
    #        x+=1
   #     i+=1

#perc = x/i
#print(x, i, perc)
#df = pd.read_csv('Fraud.csv')
#df = df[['CASE_ID', 'FRAUD', 'TOTAL_SPEND','DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS'  ]]

#print(df.head(10))
#print(df.describe(include='all'))


# Load the necessary libraries

fraud = pd.read_csv('Fraud.csv', delimiter=',')

fraud["CaseID"] = fraud["CASE_ID"]

fraud_wIndex = fraud.set_index("CaseID")

# Specify the kNN
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

# Specify the training data
trainData = fraud_wIndex[['CASE_ID', 'FRAUD', 'TOTAL_SPEND','DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
trainData.describe()

# Build nearest neighbors
#nbrs = kNNSpec.fit(trainData)
#distances, indices = nbrs.kneighbors(trainData)

# Find the nearest neighbors of these focal observations       
#focal = [[7500,15,3,127,2,2]]     

#myNeighbors = nbrs.kneighbors(focal, return_distance = False)
#print("My Neighbors = \n", myNeighbors)


# Orthonormalized the training data
x = numpy.matrix(trainData.values)

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
dvals = 1.0 / numpy.sqrt(evals)
transf = evecs * numpy.diagflat(dvals)
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)

# Specify the kNN
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(transf_x)
distances, indices = nbrs.kneighbors(transf_x)

# Find the nearest neighbors of these focal observations       
focal = [[7500,15,3,127,2,2]]    
         
        
transf_focal = focal * transf;

myNeighbors_t = nbrs.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = \n", myNeighbors_t)


#score funcion:
df 