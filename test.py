# Disable tensorflow debugging output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
from helpers import preprocess, plot
import numpy as np
import requests
import bentoml
import tensorflow_datasets as tfds
import tensorflow as tf


# Inference variables
inference_url = 'http://localhost:3000'
batch_size = 16

# Load the dataset and class names
print("Loading dataset...")
dataset, info = tfds.load('cassava', with_info=True)
class_names = info.features['label'].names + ['unknown']

# Shuffle the dataset with a buffer size equal to the number of examples in the 'validation' split
validation_dataset = dataset['validation']
buffer_size = info.splits['validation'].num_examples
shuffled_validation_dataset = validation_dataset.shuffle(buffer_size)

# Select a batch of examples from the validation dataset
batch = shuffled_validation_dataset.map(preprocess).batch(batch_size).as_numpy_iterator()
examples = next(batch)

# Convert the TensorFlow tensor to a numpy array
input_data = np.array(examples['image'])

with bentoml.SyncHTTPClient(inference_url) as client:
    print("Sending Inference Request...")
    data_array = client.predict(payload=input_data)
    print("Got Response...")

print("Predictions:", data_array)

# Convert the numpy array to tf tensor
data_tensor = tf.convert_to_tensor(np.squeeze(data_array), dtype=tf.int64)

# Plot the examples with their predictions
plot(examples, class_names, data_tensor)
