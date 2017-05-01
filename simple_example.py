from sklearn.datasets import load_iris
from sklearn import tree
import tree_extract_rule
import pandas as pd

irisdata = load_iris()


iristree = tree.DecisionTreeClassifier()
iristree = iristree.fit(irisdata.data, irisdata.target)

#the data is not a panda Dataframe, thats why convert
iris_data_pd = pd.DataFrame(irisdata.data)
iris_target_pd = pd.Series(irisdata.target, name='target')

#need featurenames:
liste = iris_data_pd.columns

#extract rules:
rules=tree_extract_rule.extract_rules(iristree, liste, iris_data_pd, iris_target_pd)

rules = pd.DataFrame.from_dict(rules)
print(rules)
for i in rules.keys():
    print('Rule: '+str(i))
    print(rules[i])
    print('\n')