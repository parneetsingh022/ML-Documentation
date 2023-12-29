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