import numpy as np
import pandas as pd


def extract_rules(tree_given, features, dataset, target_dataset, show_test_dist=False,
                  regel=None):
    """
    This function returns the the rules of the Decision Tree.
    :param tree_given: decision Tree
    :param features: please use 'features=dtrain.columns' directly before training the tree and use the list as features
    :param dataset: dataset the decisionTree got (Data) (can be test or train data) (important: Type: Dataframe)
    :param target_dataset: dataset the decisionTree got (Target) (can be test or train data)(important: Type: Dataframe)
    :param show_test_dist: Only use if the dataset is the same dataset the tree is trained.
            If this is the case 'test_class_dist' should be the same as 'class_dist' in the dictionary.
    :param regel: Name of class on which the rules point (only rules that point to special class). if None: all rules
            are printed
    you want to preserve. DONT USE THIS IF YOU ARE NOT SURE WHAT YOU ARE DOING.
    """

    if not isinstance(dataset, pd.DataFrame):
        raise Exception("dtrain has to be a Dataframe")

    if not isinstance(target_dataset, pd.Series):
        raise Exception("ttrain has to be a Dataframe")

    # features = dtrain.columns
    return_dict = {}
    list_leaf = _extract_leafs(tree_given, tree_given.classes_, regel)
    for count, leaf in enumerate(list_leaf):
        tree_path = _buildtree(tree_given, leaf)
        dist = _get_dist(tree_path, features, dataset, target_dataset, tree_given.classes_)
        dict_temp = _print_tree(tree_path, features, tree_given.classes_, dist, target_dataset)
        dict_temp = {count: dict_temp}
        if show_test_dist:
            tree_dist_train = tree_given.tree_.value[leaf].tolist()  # zur ueberpruefung anschalten
            tree_classes_list = tree_given.classes_.tolist()
            dict_temp[count]['test_class_dist'] = dict(zip(tree_classes_list, *tree_dist_train))
        return_dict.update(dict_temp)
    return return_dict


# returns the parent leaf and the value if left or right bought is used
def _parent(child_left, child_right, actual):
    if actual in child_left:
        for idx, i in enumerate(child_left):
            if i == actual:
                newvalue = idx
                tr_fl = True
    if actual in child_right:
        for idx, i in enumerate(child_right):
            if i == actual:
                newvalue = idx
                tr_fl = False
    return newvalue, tr_fl


# returns the basis structure of the rules
def _buildtree(tree_given, start):
    child_left = tree_given.tree_.children_left
    child_right = tree_given.tree_.children_right
    features = tree_given.tree_.feature
    treshold = tree_given.tree_.threshold
    # number of the feature of the first node is the class of the rule
    class_number = np.argmax(tree_given.tree_.value[start])
    tree_path = pd.DataFrame([[start, class_number, np.NaN, -5]],
                             columns=['node', 'feature', 'condition', 'true_false'])
    node = start
    while node != 0:  # going backwards in the tree to the beginning of the tree
        node, tr_fl = _parent(child_left, child_right, node)  # get the parent_leaf
        feature = features[node]
        condition = treshold[node]
        tree_temp = pd.DataFrame([[node, feature, condition, tr_fl]],
                                 columns=['node', 'feature', 'condition', 'true_false'])
        tree_path = tree_path.append(tree_temp, ignore_index=True)

    tree_path = tree_path.sort_index(axis=0, ascending=False)
    return tree_path


# returns all leafs which are in the class(rule), if rule is None all rules for all classes are returned
def _extract_leafs(tree_given, classes, rule):
    features = tree_given.tree_.feature
    list_leafs = []
    list_leafs_end = []
    for idx, i in enumerate(features):
        if i == -2:
            list_leafs.append(idx)
    if rule is not None:
        for i in list_leafs:
            if rule == classes[np.argmax(tree_given.tree_.value[i])]:
                list_leafs_end.append(i)
    else:
        list_leafs_end = list_leafs
    return list_leafs_end


# returns the distribution of the rule  mit Panda!!!!
def _get_dist(tree_path, feature_liste, ddata, tdata, classes):
    dist = pd.DataFrame(columns=['elements', 'uniques'])    # index= class
    data_ges = pd.concat([ddata, tdata], axis=1)
    for i in range(tree_path.shape[0]):     # going through all features
        if tree_path.true_false[i] == -5:
            continue    # continue if the leaf is the last one so there is no feature with value
        if tree_path.true_false[i]:
            data_ges = data_ges[data_ges[feature_liste[tree_path.feature[i]]] <= tree_path.condition[
                i]]  # delete all rows which not fullfill the criterion
        else:
            data_ges = data_ges[data_ges[feature_liste[tree_path.feature[i]]] > tree_path.condition[i]]
    for i in classes:  # going through all classes
        dist_temp = data_ges[data_ges[tdata.name] == i]  # delete if the row is not in the class
        if dist_temp.shape[0] == 0:
            dist.loc[i] = [0, 0]
        else:
            dist_temp = dist_temp.drop(tdata.name, 1)
            uniq = dist_temp.drop_duplicates()  # deletes the duplicates to get the uniqes
            dist.loc[i] = [dist_temp.shape[0], uniq.shape[0]]
    return dist


# returns a string of the rule and distribution
def _print_tree(tree_path, feature, classes, dist, ttrain):
    rule = []
    for i in range(tree_path.shape[0] - 1, 0, -1):
        if tree_path.true_false[i]:
            rule.append('If ' + str(feature[tree_path.feature[i]]) + ' <= ' + str(tree_path.condition[i]) + '\n')
        else:
            rule.append('If ' + str(feature[tree_path.feature[i]]) + ' > ' + str(tree_path.condition[i]) + '\n')
    rule = ''.join(rule)

    dist_dict = {}
    for cl in classes:
        dist_dict.update({cl: float(dist.loc[cl, 'elements'])})

    # precision
    rule_class = classes[tree_path.feature[0]]
    data_sum = dist['elements'].sum()
    data_class = np.NaN
    for i in dist.index:
        if i == str(rule_class):
            data_class = dist['elements'][i]
    if data_sum == 0:
        precision = np.NaN
    else:
        precision = float(data_class) / float(data_sum)

    # recall
    ttrain_regel = len([i for i in ttrain if i == classes[
        tree_path.feature[0]]])  # deletes all rows, which are not of the wanted (rule) class
    recall = 'Error'
    for i in dist.index:
        if i == str(rule_class):
            if ttrain_regel == 0:
                recall = np.NaN
            else:
                recall = float(dist['elements'][i]) / float(ttrain_regel)

    return {'rule': rule, 'targetclass': classes[tree_path.feature[0]], 'class_dist': dist_dict, 'precision': precision,
            'recall': recall}
