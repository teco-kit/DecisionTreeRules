## DecisionTreeRules
This modules are made to get and modify the tree rules out of a decision tree from sklearn.
There are examples how to use in the file: example.py
There are three methods implemented:
	- extract rules: 
		Return value: dictionary of rules with precision and recall in related to the used Dataset.
		Structure of the dictionary:
		{0:{'rule': If feature1 < 0.4 ..., 'targetclass': 'high' , 'class_dist': {'low': 0.0, 'high': 4.0}, 'precision': 0.9,'recall': 0.3},1: ....}		
 
		The method needs the following parameters.
    		:param tree_given: decision Tree
    		:param features: please use 'features=dtrain.columns' directly before training the tree and use the list as features
    		:param dataset: dataset the decisionTree got (Data) (can be test or train data) (important: Type: Dataframe)
    		:param target_dataset: dataset the decisionTree got (Target) (can be test or train data)(important: Type: Dataframe)
    		:param show_test_dist: Only use if the dataset is the same dataset the tree is trained. If this is the case 'test_class_dist' should be the same as 'class_dist' in the dictionary.
    		:param regel: Name of class on which the rules point (only rules that point to special class). if None: all rules are printed

		
	
		
#Hello
