import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import bentoml

# Define a service around our Model
@bentoml.service
class CassavaModel:

  def __init__(self) -> None:
    # Load the model into memory
    tf.config.experimental.set_visible_devices([], 'GPU')
    model_path = './model'
    self._model = hub.KerasLayer(model_path)

  # Logic for making predictions against our model
  @bentoml.api
  async def predict(self, payload: np.ndarray) -> np.ndarray:
    # convert payload to tf.tensor
    payload_tensor = tf.constant(payload)

    # Make predictions
    predictions = self._model(payload_tensor)
    predictions_max = tf.argmax(predictions, axis=-1)

    # convert predictions to np.ndarray
    response_data = np.array(predictions_max)

    return response_data
