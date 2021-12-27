
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_values, y_values) # Finding the best tree that fits into the training data.
print(model.predict([ [0.2, 0.8], [0.5, 0.4] ])) # return an array of predictions, on prediction for each input array.

```

To set hyperparameters, add parameters at the definition step.

`max_depth`: The maximum number of levels in the tree.
`min_samples_leaf`: The minimum number of samples allowed in a leaf.
`min_samples_split`: The minimum number of samples required to split an internal node.

```python
model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10)
```
