import tensorflow as tf


def feed_forward(
        x,
        num_hiddens,
        activation=None,
        reuse=False,
):
    with tf.variable_scope('feed-forward2', reuse=reuse):
        ff = tf.layers.dense(x, num_hiddens, activation=activation, reuse=reuse)
    return ff


def linear(
        x,
        num_hiddens=None,
        reuse=False,
):
    if num_hiddens is None:
        num_hiddens = x.get_shape().as_list()[-1]
    # with tf.variable_scope('linear'):
    linear_layer = tf.layers.dense(x, num_hiddens)
    return linear_layer


def dropout(
        x,
        is_training,
        rate=0.2,
):
    return tf.layers.dropout(x, rate, training=tf.convert_to_tensor(is_training))


def residual(
        x_in,
        x_out,
        reuse=False,
):
    with tf.variable_scope('residual', reuse=reuse):
