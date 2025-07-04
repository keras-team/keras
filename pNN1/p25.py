import cd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
# 1. Prepare Input Data with Pandas (might be useful for hardcoding the eddy test points or something)
data = {'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3'
        'target': np.random.randint(0, 2, 100)}
"""


def create_ESC_data(
    measurement_instances, measurements_per_instance, classes_of_outcomes
):  # Creates a genaric random data set for ESC where measurements_instance is the object that represents a given serial number*coating instance

    # classes of outcomes is basically always gonna be 2 for crack / not cracked

    np.random.seed(0)

    # redminder to use this later when you have to actually feed it data. Gonna have to normalize each category
    # it's gonna be important that I normalize everything within its 'measurement_per_instance' class. Like I probably shouldn't compare raw numbers of eddy test with raw numbers from spray nozel cuz they are a couple of orders of magnitude off and will probably blow up computer
    input_data_eddy_top = []
    outcome_data = []
    for measurement in range(
        measurement_instances
    ):  # just a cheap way to iterate over a range by input of an integer

        np.random.seed(measurement)
        input_data_eddy_top.append(
            40 + 2.5 * np.random.randn(measurements_per_instance)
        )
        outcome_data.append(
            np.random.randint(classes_of_outcomes) + 1
        )  # ranom number from 1 to classes_of_outcomes

        # print(input_data_eddy_top)
        # simple for now just a random normal distribution of 40 micron measurements with 2.5 stdrd deviation of

    # normalized input data = input_data_eddy_top =
    # here is where I would have to break it down

    # input_data_eddy_side =

    # input_data_bot_eddy =

    # post_blast_roughness =

    X = input_data_eddy_top
    # print(X)
    y = outcome_data

    return X, y


data_shape = [5, 5, 2]
# print(create_ESC_data(100,44,2))
data, target = (
    create_ESC_data(data_shape[0], data_shape[1], data_shape[2])[0],
    create_ESC_data(data_shape[0], data_shape[1], data_shape[2])[1],
)  # 100,44,2 matrix with 2 'targets' or outcomes where entry 0 is 100,44. Creates set of X variables
# print(data)

# target =  # 100,1 matrix of 0 or 1. Creates y variable
# print(target)


# random seed is 0
# data = cd.create_data(100,10)[0] # data is a matrix with 100 rows of 2 caretesian coordinate points, where r

# target = cd.create_data(100,10)[1]# target is a 100 row matrix with 0 or 1 as possible classes
# print(target)
df = pd.DataFrame(data)
print(df.loc[0])
print(data[0])
# X = df[["feature1", "feature2"]].values

X = tf.convert_to_tensor(
    np.array(data, dtype=np.int32)
)  # tf.data.Dataset.from_tensor_slices(data, data[1])
print(X)
# print('shape is: ', X.shape[1])
y = target


# 2. Define the 2-Layer Network Architecture
model = keras.Sequential(
    [
        layers.Dense(units=5, activation="relu", input_shape=(5, 5)),
        layers.Dense(
            units=2, activation="sigmoid"
        ),  # Output layer for binary classification
    ]
)

# print(L1.compute_output_shape(input_shape = (X.shape[1])))
# L2 =

# L2 = layers.Dense(units=1, activation='softmax', input_shape = L1.compute_output_shape((X.shape[1]),)   ) # Intermediate

# L3 = layers.Dense(units=1, activation='sigmoid') # Output layer for binary classification


# 2. Define the 3-Layer Network Architecture
# model = keras.Sequential([ layers.Dense(units=10, activation='relu', input_shape = (X.shape[1])), layers.Dense(units=1, activation='softmax')])    # Intermediate # Hidden layer, L2, L3 ])


# 3. Train the Network
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
model.fit(X, y, epochs=100, batch_size=44)

# 4. Evaluate and Predict
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

new_data = pd.DataFrame({"feature1": [0.1, 0.9], "feature2": [0.2, 0.8]})
predictions = model.predict(new_data.to_numpy())
print(f"Predictions: {predictions}")
