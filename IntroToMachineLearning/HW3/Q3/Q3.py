import pandas as pd
import statsmodels.api as stats
import numpy
import scipy

df = pd.read_csv("sample_v10.csv")

col_y = df["y"].unique()

print(col_y)
freqy=[0,0,0]
for i in df["y"]:
    if i == 3:
        freqy[0]+=1
    if i == 2:
        freqy[1]+=1
    if i == 1:
        freqy[2]+=1
        
print (freqy)


# Specify Origin as a categorical variable
Origin = df['y'].astype('category')
y = Origin
y_category = y.cat.categories

# Backward Selection
# Consider Model 0 is y = Intercept + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
DriveTrain = df[['x1']].astype('category')
X = df[['x1']]
X = X.join(df[['x2']])
X = X.join(df[['x3']])
X = X.join(df[['x4']])
X = X.join(df[['x5']])
X = X.join(df[['x6']])
X = X.join(df[['x7']])
X = X.join(df[['x8']])
X = X.join(df[['x9']])
X = X.join(df[['x10']])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)


print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)




chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
ll=[LLK1]

for i in chars :

    #calculates the DF and LLK for the var about to be removed
    DF_prev = df[['x'+i]].astype('category')
    X = stats.add_constant(X, prepend=True)
    DF_prev = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK_prev = logit.loglike(thisParameter.values)

    

    # drop the value
    X.drop('x' + i, axis = 1, inplace=True)
    
    #calculates the DF and LLK for the current var
    DF = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)
    
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK = logit.loglike(thisParameter.values)
    
    ll.append(LLK)
    
    #calculates the deviance
    
    Deviance = 2 * (LLK - LLK_prev)
    DF_now = DF - DF_prev
    pValue = scipy.stats.chi2.sf(Deviance, DF_now)
    
    print("Current DF value: " + str(DF_now))
    print("Current Deviance: " + str(Deviance))
    print("Current pValue: " + str(pValue))
    
    #print(thisFit.summary())
    print()
    print()
    print("-----------Model "+i+"-------------")
    print("x"+ i + " removed")
    print()
    print("Model Log-Likelihood Value =", LLK)
    print("Number of Free Parameters =", DF)
    print("Deviance (Statistic, DF, Significance)", Deviance, DF_now, pValue)
          

#AIC = -2/N * LL + 2 * k/N
#Where N is the number of examples in the training dataset, LL is the log-likelihood
#of the model on the training dataset,
#and k is the number of parameters in the model.

#BIC = -2 * LL + log(N) * k
#Where log() has the base-e called the natural logarithm, LL is the log-likelihood of the model,
#N is the number of examples in the training dataset, and k is the number of parameters in the model.


#3e

#print(ll)

print()
print()
print("Model 0")
print("all elements")
AIC = -2/1.13 * ll[0] + 2 *10/1.13
BIC = -2 * ll[0] + numpy.log(1.13) * 10

print("ACI: " + str(AIC))
print("BIC: " + str(BIC))
    
for i in range(1,11):
    print()
    print("Model "+ str(i))
    print("x"+str(i) + "Removed")
    AIC = -2/1.13 * ll[i] + 2 *10/1.13
    BIC = -2 * ll[i] + numpy.log(1.13) * 10

    print("ACI: " + str(AIC))
    print("BIC: " + str(BIC))
    

    