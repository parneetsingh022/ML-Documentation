# Pandas

## Loading a csv file
```python
import pandas as pd
df = pd.read_csv("path/to/file.csv")
```

## One Hot Encoding
```python
df_one_hot = pd.get_dummies(df)
```

## Dropping columns
```python
X = df.drop_column("column_name", axis=1)
```

## Getting a specific column
```python
y = df['column_name']
```

## Plotting the loss curve
We can plot the loss curve in the following way:
```python
history = model.fit(.....)

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epoch")
```
## Plotting hist plot
```python
X[['column1', 'column2', ...]].hist()
```

## Decision Plot
```python
import numpy as np

def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary created by a model predicting on X.
  """

  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

  xx, yy = np.meshgrid(
      np.linspace(x_min, x_max,100),
      np.linspace(y_min, y_max,100))

  # create X value (we're going to make predictions on these)
  x_in  = np.c_[xx.ravel(), yy.ravel()] # stack 2D array together

  # Make predictions
  y_pred = model.predict(x_in)

  # Check for multi-class
  if (len(y_pred[0])> 1):
    print("doing multiclass classification")
    # we have to reshape our prediction to get them ready for plotting

    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)

  # plot decision boundary
  plt.contourf(xx,yy,y_pred,cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
```