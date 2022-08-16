import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC

spiral_df = pd.read_csv("SpiralWithCluster.csv")


#target variables
target =  spiral_df.SpectralCluster
predictors = spiral_df[["x","y"]]


#The svm model
svm_clf = SVC(kernel = "linear", random_state=20210325, decision_function_shape='ovr',max_iter=-1,probability = True)
svm_clf.fit(predictors,target)
spiral_df["SVMClusters"] = svm_clf.predict(predictors)
svm_pred = svm_clf.predict(predictors)

#misclassification rate
def misclassification_rate(pred_proba,target):
    threshold = 0.50
    counter = 0
    answer =[]
    for i in pred_proba:
        if i > threshold:
            answer.append(1)
        else:
            answer.append(0)
    for j in range(len(answer)):
        if answer[j] != target[j]:
            counter += 1
    return (counter/len(answer))  

pred_proba_result = pd.DataFrame(data=svm_clf.predict_proba(predictors),columns = ["A0","A1"])
pred_proba_result["A0"]
missclass = misclassification_rate(pred_proba_result["A1"],target)
print("The miscalssification rate before transformation is",missclass)



#creating The line to seperate The hyper plane
w = svm_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm_clf.intercept_[0]) / w[1]


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(xx, yy, 'k--')


for i in range(2):
    subdata = spiral_df[spiral_df["SVMClusters"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot according to Cluster Values")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.show()


# equation to seperate The hyper plane
print ('The equation of The seperating hyperplane is')
print (svm_clf.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")


#expressing The polar coordinates

def customArcTan (z):
    Theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (Theta)

trainData = pd.DataFrame(columns = ["radius","Theta"])
trainData['radius'] = np.sqrt(spiral_df['x']**2 + spiral_df['y']**2)
trainData['Theta'] = (np.arctan2(spiral_df['y'], spiral_df['x'])).apply(customArcTan)

trainData['class']=spiral_df["SpectralCluster"]
print(trainData.head())

#ploting The Theta-coordinate against The radius-coordinate 
colur = ['red','blue']
for i in range(2):
    subdata = trainData[trainData["class"]==i]
    plt.scatter(subdata.radius,subdata.Theta,label = (i),c = carray[i])
    
plt.title("Scatterplot of Polar Co-ordinates")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.grid()
plt.show()

#creating Group

x = trainData["radius"]
y = trainData['Theta'].apply(customArcTan)
svm_dataframe = pd.DataFrame(columns = ['Radius','Theta'])
svm_dataframe['Radius'] = x
svm_dataframe['Theta'] = y

group = []

for i in range(len(x)):
    if x[i] < 1.5 and y[i]>6:
        group.append(0)
        
    elif x[i] < 2.5 and y[i]>3 :
        group.append(1)
    
    elif 2.75 > x[i]>2.5 and y[i]>5:
        group.append(1)
        
    elif 2.5<x[i]<3 and 2<y[i]<4:
        group.append(2)      
        
    elif x[i]> 2.5 and y[i]<3.1:
        group.append(3)
        
    elif x[i] < 4:
        group.append(2)
        

#plotting The Theta-coordinate against The radius-coordinate in a scater plot, with new collor codes
svm_dataframe['Class'] = group
colors = ['red','blue','green','black']
for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = colors[i],label=i)
plt.grid()
plt.title("Scatterplot with four Groups")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.show()


#SVM to classify class 0 and class 1
svm_1 = SVC(kernel = "linear", random_state=20210325, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 0]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 1])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 2)
yy = a * xx - (svm_1.intercept_[0])/w[1] 

print ('The equation of The hypercurve for SVM 0 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h0_xx = xx * np.cos(yy[:])
h0_yy = xx * np.sin(yy[:])

carray=['red','blue','green','black']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#Plot Ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to classify class 1 and class 2
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 1]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 2])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 4)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('The equation of The hypercurve for SVM 1 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h1_xx = xx * np.cos(yy[:])
h1_yy = xx * np.sin(yy[:])


#Plot Ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to. classify class 2 and class 3
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 2]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 3])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(2, 4.5)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('The equation of The hypercurve for SVM 2 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h2_xx = xx * np.cos(yy[:])
h2_yy = xx * np.sin(yy[:])


#Plot Ther hyperplane
plt.plot(xx, yy, 'k--')


for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = carray[i],label=i)
plt.xlabel("Radius")
plt.ylabel("Theta Co-Ordinate")
plt.title("Scatterplot of The polar co-ordinates with 4 diffrent classes seperated by 3 hyperplanes")
plt.show()

carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    subdata = spiral_df[spiral_df["SpectralCluster"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of The cartesian co-ordinates seperated by The hypercurve")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.show()



















