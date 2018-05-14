from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn import tree
import pandas as pd
import tree_extract_rule

boston = load_boston()  # Load Dataset

boston_target = pd.Series(boston.target, name='target')  # target_data

boston_class = pd.Series()  # creat Dataserie for classes
for idx, i in enumerate(boston_target):
    if i <= 25:
        boston_class = boston_class.append(pd.Series(['low'], index=[idx]))
    if i > 25:
        boston_class = boston_class.append(pd.Series(['high'], index=[idx]))

boston_class.name = 'target'  # Dataserie with classes
boston_data = pd.DataFrame(boston.data, columns=boston.feature_names)  # create DataFrame out of Dataset

liste = boston_data.columns  # read the column names for rule_extraction
blf = tree.DecisionTreeClassifier()  # create the tree class_weight='balanced'
blf = blf.fit(boston_data, boston_class)  # train the tree

rules = tree_extract_rule.extract_rules(blf, liste, boston_data, boston_class)  # extract rules ,target_class='high'

r = pd.DataFrame.from_dict(rules)
print(r)

# adding target column to dataset
boston_data['target'] = boston_class

# get elemnts of dataset of one certain rule:
elements = tree_extract_rule.extract_elements_of_rule(boston_data,
                                                      'If RM <= 6.54549980164\nIf DIS <= 1.33920001984\nIf LSTAT > 17.7350006104\n')
print(elements)

# cut rules with 'LSTAT' in the variable and max_precision of 0.8 and min_recall of 0.05
cutted_rules = tree_extract_rule.cut_tree_rules(rules, boston_data, 'target', cut_feature_str='LSTAT',
                                               max_precision=0.8, min_recall=0.05)

for i in cutted_rules.keys():
    print(cutted_rules[i])
