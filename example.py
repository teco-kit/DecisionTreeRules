from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import tree_extract_rule
from sklearn.tree import export_graphviz


boston = load_boston() #Load Dataset

boston_target = pd.Series(boston.target, name='target')        #target_data

boston_class = pd.Series()                                     #creat Dataserie for classes
for idx, i in enumerate(boston_target):
    if i <= 25:
        boston_class = boston_class.append(pd.Series(['low'], index=[idx]))
    if i > 25:
        boston_class = boston_class.append(pd.Series(['high'], index=[idx]))

boston_class.name = 'target'                                    #Dataserie with classes
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)      #create DataFrame out of Dataset

liste=boston_data.columns                                       #read the column names for rule_extraction
blf = tree.DecisionTreeClassifier()                             #create the tree
blf = blf.fit(boston_data, boston_class)                        #train the tree



rules = tree_extract_rule.extract_rules(blf, liste,boston_data, boston_class)   #extract rules

#print(rules[0]['class_dist'])
#print(rules[0]['test_class_dist'])#print example

print(rules)
boston_data['target']=boston_class

nrules=tree_extract_rule.cut_tree_rules(rules, boston_data,'target',max_precision=0.8, min_recall=0.05)

export_graphviz(blf,feature_names=liste, out_file='tree.dot')

for i in nrules.keys():
    print(nrules[i])
