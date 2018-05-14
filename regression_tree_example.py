import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import tree_extract_rule
import pandas as pd

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# extract rules
# convert to pandas
X = pd.DataFrame(X)
y = pd.Series(y)

# extract rule
rules = tree_extract_rule.extract_rules(regr_1, X.columns, X, y)

r = pd.DataFrame.from_dict(rules)
print(r)

rule1 = rules[0]['rule']   #extract rule
print(rule1)
X1 = X.copy()       #copy X data and add y
X1['target'] = y
extract_elements = tree_extract_rule.extract_elements_of_rule(X1, rule1)  # extract the data
print(extract_elements)
print(extract_elements.apply(lambda x: [x.mean(),x.std()], axis=0)) # calculate the mean and std on the variables of the Dataset. if necessary the

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# extract rules
