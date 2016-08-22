import numpy as np
import pandas as pd
from sklearn import tree


# Funktion zum Aufrufen uber python und ausgeben der Regeln
def extract_rules(tree_given, classes, features, dtrain, ttrain,
                  regel=None):
    """
    This function returns the the rules of the Decision Tree.
    :param tree_given: decision Tree
    :param classes: list of classes the decision tree points on.
    :param features: list of name of features (same lengh as columns in Data)
    :param dtrain: dataset the decisionTree got (Data)
    :param ttrain: dataset the decisionTree got (Target)
    :param regel: Name of class on which the rules point (only rules that point to special class). if None: all rules are printed

    you want to preserve. DONT USE THIS IF YOU ARE NOT SURE WHAT YOU ARE DOING.
    """

    rules_box = []
    list_leaf = _extract_leafs(tree_given, classes, regel)
    for leaf in list_leaf:
        tree_path, valu = _buildtree(tree_given, leaf)
        dist = _get_dist(tree_path, dtrain, ttrain, classes)
        text = _print_tree(tree_path, features, classes, dist)
        print('-' * 50 + '\n')
        rules_box.extend('-' * 50 + '\n')
        print(text)
        rules_box.extend(text)
    return


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


# erstellen der Grundstruktur der Regeln
def _buildtree(tree_given, start):
    child_left = tree_given.tree_.children_left
    child_right = tree_given.tree_.children_right
    features = tree_given.tree_.feature
    treshold = tree_given.tree_.threshold
    # feature von anfangszahl ist die endklasse
    class_number = np.argmax(tree_given.tree_.value[start])
    tree_path = pd.DataFrame([[start, class_number, np.NaN, -5]],
                             columns=['node', 'feature', 'condition', 'true_false'])
    node = start
    while node != 0:
        node, tr_fl = _parent(child_left, child_right, node)
        node_temp = node
        name_temp = features[node]
        condition_temp = treshold[node]
        tr_fl_temp = tr_fl
        tree_temp = pd.DataFrame([[node_temp, name_temp, condition_temp, tr_fl_temp]],
                                 columns=['node', 'feature', 'condition', 'true_false'])
        tree_path = tree_path.append(tree_temp, ignore_index=True)

    tree_path = tree_path.sort_index(axis=0, ascending=False)
    return tree_path, tree_given.tree_.value[start]


# Filter fur die Blatter:
def _extract_leafs(tree_given, classes, regel):
    features = tree_given.tree_.feature
    list_leafs = []
    list_leafs_end = []
    for idx, i in enumerate(features):
        if i == -2:
            list_leafs.append(idx)
    if regel != None:
        for i in list_leafs:
            if regel == classes[np.argmax(tree_given.tree_.value[i])]:
                list_leafs_end.append(i)
    else:
        list_leafs_end = list_leafs
    return list_leafs_end


# erstellen der Distribution für den Datensatz mit dem auch der Tree erstellt wurde
def _get_dist(tree_path, ddata, tdata, classes):
    dist = [['Class', 'Elements', 'Uniques']]
    tdata = np.array([tdata]).T
    data_ges = np.append(ddata, tdata, axis=1)
    for i in range(tree_path.shape[0]):
        if tree_path.true_false[i] == -5:
            continue
        if tree_path.true_false[i]:
            data_ges = data_ges[data_ges[:, tree_path.feature[i]] <= tree_path.condition[i]]
        else:
            data_ges = data_ges[data_ges[:, tree_path.feature[i]] > tree_path.condition[i]]
    for i in classes:
        dist_temp = data_ges[data_ges[:, -1] == i]
        if dist_temp.shape[0] == 0:
            continue
        else:
            dist_temp = np.delete(dist_temp, -1, 1)
            dist_temp_pd = pd.DataFrame(dist_temp)  # transform to a pandas Dataframe
            uniq = dist_temp_pd.drop_duplicates()  # deletes the duplicates to get the uniqes
            dist = np.append(dist, [[i, dist_temp.shape[0], uniq.shape[0]]], axis=0)
    return dist


# Ausgabe des Baumes

def _print_tree(tree_path, feature, classes, dist):
    text = ['Class: ' + str(classes[tree_path.feature[0]]) + '\n' + '\n']
    for i in range(tree_path.shape[0] - 1, 0, -1):
        if tree_path.true_false[i]:
            text.append('If ' + str(feature[tree_path.feature[i]]) + ' <= ' + str(tree_path.condition[i]) + '\n')
        else:
            text.append('If ' + str(feature[tree_path.feature[i]]) + ' > ' + str(tree_path.condition[i]) + '\n')
    text.append('\n Distribution:\n')
    for i in range(dist.shape[0]):
        text.append(dist[i, 0] + ':\t' + str(dist[i, 1]) + ', ' + str(dist[i, 2]) + '\n')

    text.append('\n')
    return ''.join(text)
