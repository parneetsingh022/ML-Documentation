# Scikit-Learn

## Train-Test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Normalization
```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["column1", "column2", "column3"]), # turn all values in these columns between zero and ones
    (OneHotEncoder(handle_unknown="ignore"), ["column1", "column2", "column3"])
)

# Create X & y values
X = df.drop("charges", axis=1)
y = df["charges"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(X_train)

# transform training and test data with normalization (MinMaxScaler and OneHotEncoding)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)
```