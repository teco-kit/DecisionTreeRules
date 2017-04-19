# DecisionTreeRules
This modules are made to get and modify the tree rules out of a decision tree from sklearn.
There are examples how to use in the file: example.py

## extract_rules
This methode is to extract the rules out of a decision tree. It is possible to calculate the precision and recall for other Dataset then the training dataset.
Return value: dictionary of rules with precision and recall in related to the used Dataset.

Structure of the dictionary:  
    {0:{'rule': If feature1 < 0.4 ..., 'targetclass': 'high' , 'class_dist': {'low': 0.0, 'high': 4.0}, 'precision': 0.9,'recall': 0.3},1: ....}

The methode needs the following parameters:  
    +param tree_given: decision Tree  
    +param features: please use 'features=dtrain.columns' directly before training the tree and use the list as features  
    +param dataset: dataset the decisionTree got (Data) (can be test or train data) (important: Type: Dataframe)  
    +param target_dataset: dataset the decisionTree got (Target) (can be test or train data)(important: Type: Dataframe)  
    +param show_test_dist: Only use if the dataset is the same dataset the tree is trained. If this is the case 'test_class_dist' should be the same as 'class_dist' in the dictionary.  
    +param regel: Name of class on which the rules point (only rules that point to special class). if None: all rules are printed  

		
	
		
## extract_elements_of_rulebelo
This methode is to extract certain elements of a dataset (format: pandas.DataFrame) belonging to a given rule.  
The methode returns a pandas.DataFrame with all data belonging to the rule.  

The methode needs the following parameters:  
    +param data: dataset to search in (panda)
    +param rule: string, rule you want to use

## cut_tree_rules
This methode is to cut/modify the tree rules. There are three possibilities to cut/modify the tree. Please consider that the methode first filters for feature_strings, than precision and then recall values.
    +cuting at a certain variable
    +cuting at a certain precision
    +cuting at a certain recall

The methode returns the type of dictionary as the extract_rules methode.  

It needs the following paramters:  
    +param target_variabel_name: string of the target feature
    +param cut_feature_str: feature name of the feature to cut.
    +param data: dataset to search in (panda)
    +param dict_of_rules_to_cut: list of strings, rule you want to use
    +param max_precision: float value: if the precision is higher than this value the tree is cutted
    +param min_recall: float value: if the recall is gets smaller than this value the tree is cutted before

### Questions or annotations?
Please contact: gregor.schindler@student.kit.edu

