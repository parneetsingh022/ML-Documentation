# Tensorflow
## Building a TensorFlow model
Here we have used `SGD` (Stochastic Gradient Descent) optimizer and `mae` (mean absolute error) loss function. We can also use `Adam` optimizer i.e. `tf.keras.optimize.Adam(learning_rate=0.001)`
```python
import tensorflow as tf
# Set random seed to have reproducibility
tf.random.set_seed(42)

# 1. Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu"),
    #... other required layers
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # sgd is short for stochastic gradient descent
              metrics=["mae","accuracy"])

#3. Fit the model
model.fit(X_tens_converted,y, epochs=5)
```

### Getting model summery
We can get details about the model using `model.summary()` function. It will provide the list of total
parameters and parametrs in each layer of the neural network.

```python
model.summary()
```

### Evaluating the model
```python
model.evaluate(X_test,y_test)
```

### Getting a prediction from a model

```python
model.predict([6])
```

### Saving the model
#### SavedModel format
The SavedModel format is a way to serialize models. Models saved in this format can be restored using `tf.keras.models.load_model` and are compatible with TensorFlow Serving.
```python
# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')
```

#### HDF5 format
Keras provides a basic legacy high-level save format using the HDF5 standard.
```python
# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')
```

### Loading the model
Model can be loaded from both SavedModel and HDF5 format
```python
#loading from SavedModel format
from_saved_model = tf.keras.models.load_model('saved_model/my_model')
#loading from HDF5 format
from_h5 = tf.keras.models.load_model('my_model.h5')
```

## Visualizing the model
We can visualize the model using `plot_model`

```python
from tensorflow.keras.utils import plot_model

plot_model(model=model,show_shapes=True)
```

## Finding the error
### Mean Absolute Error
We can write following function which will give us the `mean absolute error`.
```python
def mae(y_true,y_pred):
    return tf.metrics.mean_absolute_error(
      y_true=y_test,
      y_pred=tf.squeeze(y_pred)
    )
```

### Mean Squared Error
We can write following function which will give us `the mean squared error`.
```python
def mse(y_true,y_pred):
    return tf.metrics.mean_squared_error(
      y_true=y_test,
      y_pred=tf.squeeze(y_pred)
    )
```