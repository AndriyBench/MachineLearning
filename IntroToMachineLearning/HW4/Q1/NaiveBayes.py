import itertools
import numpy
import pandas
import sklearn.naive_bayes as naive_bayes

# Set some options for printing all the columns
pandas.set_option('precision', 13)

# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

# Specify the roles
feature = ['group_size', 'homeowner', 'married_couple']
target = 'insurance'

# Read the Excel file
likelihood = pandas.read_csv('Purchase_Likelihood.csv',
                              usecols = feature + [target])
likelihood = likelihood.dropna()

# Look at the row distribution
print(likelihood.groupby(target).size())

for pred in feature:
    RowWithColumn(rowVar = likelihood[target], columnVar = likelihood[pred], show = 'ROW')

print( )
print( )
print( )
print( )

# Make the binary features take values 0 and 1 (was 2=No and 1=Yes)
likelihood[feature] = 2 - likelihood[feature]

xTrain = likelihood[feature].astype('category')
yTrain = likelihood[target].astype('category')

_objNB = naive_bayes.BernoulliNB(alpha = 1.e-10)
thisFit = _objNB.fit(xTrain, yTrain)

print('Probability of each class')
print(numpy.exp(thisFit.class_log_prior_))

print('Empirical probability of features given a class, P(x_i|y)')
print(numpy.exp(thisFit.feature_log_prob_))

print('Number of samples encountered for each class during fitting')
print(thisFit.class_count_)

print('Number of samples encountered for each (class, feature) during fitting')
print(thisFit.feature_count_)

yTrain_predProb = _objNB.predict_proba(xTrain)

