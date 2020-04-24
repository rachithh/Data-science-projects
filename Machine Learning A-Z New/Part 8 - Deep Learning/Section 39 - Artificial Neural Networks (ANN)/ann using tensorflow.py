# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset = dataset.rename(columns={"RowNumber": "ID"})

# Put all the exited customer class in a separate dataset.
exited_df = dataset.loc[dataset['Exited'] == 1]

#Randomly select len(exited_df) observations from the non-exited customers (majority class)
non_exited_df = dataset.loc[dataset['Exited'] == 0].sample(n=2037,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([exited_df, non_exited_df])

#shuffle the dataset
shuffled_df = normalized_df.sample(frac=1,random_state=4)

# The inputs are all columns in the csv, except for the first one [:,0]
# (which is just the arbitrary customer IDs that bear no useful information),
# and the last one [:,-1] (which is our targets)
unscaled_inputs_all = shuffled_df.iloc[:,3:13]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = shuffled_df.iloc[:,13]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
unscaled_inputs_all.iloc[:, 1] = labelencoder_X_1.fit_transform(unscaled_inputs_all.iloc[:, 1])
labelencoder_X_2 = LabelEncoder()
unscaled_inputs_all.iloc[:, 2] = labelencoder_X_2.fit_transform(unscaled_inputs_all.iloc[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
'''unscaled_inputs_all = onehotencoder.fit_transform(unscaled_inputs_all).toarray()
unscaled_inputs_all = unscaled_inputs_all[:, 1:]'''

unscaled_inputs_all = preprocessing.scale(unscaled_inputs_all)

# Count the total number of samples
samples_count = unscaled_inputs_all.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

# The 'test' dataset contains all remaining data.
train_inputs = unscaled_inputs_all[:train_samples_count]
train_targets = targets_all[:train_samples_count]

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
validation_inputs = unscaled_inputs_all[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = targets_all[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
test_inputs = unscaled_inputs_all[train_samples_count+validation_samples_count:]
test_targets = targets_all[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

# Save the three datasets in *.npz.
# In the next lesson, you will see that it is extremely valuable to name them in such a coherent way!

np.savez('Churn_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Churn_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Churn_data_test', inputs=test_inputs, targets=test_targets)

# let's create a temporary variable npz, where we will store each of the three Audiobooks datasets
npz = np.load('Churn_data_train.npz')

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('Churn_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('Churn_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# Set the input and output sizes
input_size = 13
output_size = 1
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 6

#define how the model will look like
model = tf.keras.Sequential([
        # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'), #adding first hidden layer
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'), #adding second hidden layer
    # the final layer is no different, we just make sure to activate it with sigmoid
    tf.keras.layers.Dense(output_size,activation='sigmoid') #output layer
    ])

### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# set the batch size
batch_size = 10

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
#early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
early_stopping = tf.keras.callbacks.EarlyStopping()

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

# Predicting the Test set results
test_pred = model.predict(test_inputs)
test_pred = (test_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_targets, test_pred)