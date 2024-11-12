import tensorflow as tf

def weighted_binary_crossentropy(beta=0.9):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        weights = (beta * y_true + (1.0 - beta) * (1.0 - y_true))
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        weighted_bce = weights * bce
        return tf.reduce_mean(weighted_bce)
    return loss
