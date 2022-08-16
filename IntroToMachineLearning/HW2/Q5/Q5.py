import pandas as pd
import csv
unq = []
kitemset = {}
counter = 0;
groceries = pd.read_csv('Groceries.csv', delimiter=',')

#kitemset.update({1 : ["hello"]})
#print(kitemset)
#kitemset[1].append("bitches")
#if(kitemset == 1):
#    kitemset[1].append("bitches")
#print(kitemset)

for i in range(0, 43366):
    if((int(groceries['Customer'].values[i]) == int(groceries['Customer'].values[i+1])) or int(groceries['Customer'].values[i]) < int(groceries['Customer'].values[i+1]) ):
        kitemset.update({int(groceries['Customer'].values[i]) : []})

for j in range(0,43367):
    if(int(groceries['Customer'].values[j]) in kitemset):
        kitemset[int(groceries['Customer'].values[j])].append(str(groceries['Item'].values[j]))
        

for k in range(1, len(kitemset)):
    print()
    print()
    print("-------------------------------", kitemset[k] ,"---------------------------------------")
    counter = 0
    for l in range(1, len(kitemset)):
        if((k!=l) and kitemset[k] == kitemset[l]):
            print()
            print()
            print("MATCH:")
            print(kitemset[k])
            print(kitemset[l])
            counter+=1
            print(counter)   
 
        
#print(kitemset)

#print("Total Itemsets: ", 9835 )

#unq = set(kitemset.values())

for k in unq:
    print(k)


#unq = groceries.Item.unique()



#for j in range(0, len(unq)):
#    counter = 0
#    for i in range(0, 43367):
        #if(str(groceries['Item'].values[i]) == str(unq[j])):
            #counter+=1
#    kitemset[unq[j]] = counter

#print(kitemset)

#max_key = max(kitemset, key = kitemset.get)
#for i in kitemset:
#    counter+=1

#print("Total Itemsets: ", counter)
#print("Largest kitemset value: ", max_key, " - " ,kitemset[max_key])

    #print(groceries[x])


#noela@createandlearn.us
