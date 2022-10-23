import tensorflow as tf
import numpy as np

def user_behaviour(action_len):
    action = tf.keras.layers.Input(shape=(None, 1+action_len,))
    #user = tf.keras.layers.Input(shape=(1, 64,))

    #concatted = tf.keras.layers.Concatenate()([action, user])
    out = tf.keras.layers.LSTM(action_len, return_sequences=True)(action)
    out = tf.keras.layers.LSTM(action_len * 2, return_sequences=True)(out)
    out = tf.keras.layers.LSTM(action_len, return_sequences=True)(out)

    model = tf.keras.Model(inputs = [action], outputs = out)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    model.compile(loss = loss , optimizer = "adam")
    return model

if __name__ == "__main__":
    model = user_behaviour(25)
    tf.keras.utils.plot_model(
    model,
    to_file='model.png'
    )
