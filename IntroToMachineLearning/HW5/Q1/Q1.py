import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.metrics as metrics


trainData = pd.read_csv('../WineQuality_Train.csv', delimiter=',')
testData = pd.read_csv('WineQuality_Test.csv', delimiter = ',')
nObs = trainData.shape[0]
tObs = testData.shape[0]

x_train = trainData[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
x_train = trainData[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
y_train = trainData['quality_grp']

x_test = testData[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
y_test = testData['quality_grp']



# PART A
w_train = np.full(nObs, 1.0)
accuracy = np.zeros(50)
ensemblePredProb = np.zeros((nObs, 2))

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
 
print('Accuracy = ', accuracy)
print('Missclass rate = ', 1-accuracy)

dot_data = tree.export_graphviz(treeFit, out_file = None, impurity = True, filled = True, feature_names = x_train.columns, class_names = ['0','1'])
graph = graphviz.Source(dot_data)
#graph.view()




# PART B/C

# Build a classification tree on the training partition
w_train = np.full(nObs, 1.0)
accuracy = np.zeros(50)
misclass=np.zeros(50)
ensemblePredProb = np.zeros((nObs, 2))

for iter in range(50):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    ensemblePredProb += accuracy[iter] * treePredProb
    
    if (abs(1.0 - accuracy[iter]) < 0.0000001):
        break
     
    
    misclass[iter]= 1- accuracy[iter]
    # Update the weights
    eventError = np.where(y_train == 1, (1 - treePredProb[:,1]), (0 - treePredProb[:,1]))
    predClass = np.where(treePredProb[:,1] >= 0.2, 1, 0)
    w_train = np.where(predClass != y_train, 2+np.abs(eventError), np.abs(eventError))
    
ensemblePredProb /= np.sum(accuracy)

trainData['predCluster'] = np.where(ensemblePredProb[:,1] >= 0.5,1,0)

print(accuracy)   

print(misclass)




# PART D

w_test = np.full(tObs, 1.0)
accuracy = np.zeros(50)
misclassification = np.zeros(50)
ensemblePredProb = np.zeros((nObs, 2))
for iter in range(50):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    misclassification[iter] = 1 - accuracy[iter] 
    ensemblePredProb += accuracy[iter] * treePredProb        
    if (abs(1.0 - accuracy[iter]) < 0.0000001):
        break    
    # Update the weights
    eventError = np.where(y_train == 1, (1 - treePredProb[:,1]), treePredProb[:,1])
    predClass = np.where(treePredProb[:,1] >= 0.2, 1, 0)
    w_train = np.where(predClass != y_train, 2 + np.abs(eventError), np.abs(eventError))
AUC = metrics.roc_auc_score(y_train, treePredProb[:,1])
print('AUC = ', AUC)



# PART E

bagPredProb = treeFit.predict_proba(x_test)
AUC_test = metrics.roc_auc_score(y_test, bagPredProb[:,1])
print('AUC = ', AUC_test)

testData['predict_proba_group_0'] = bagPredProb[:, 0]
testData['predict_proba_group_1'] = bagPredProb[:, 1]

testData.boxplot(column='predict_proba_group_1', by='quality_grp')
plt.show()



































