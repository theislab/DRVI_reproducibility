"""
Dense / batch-norm / dropout without tf.compat.v1.layers (broken when TF ships Keras 3).
Uses only tf.compat.v1.get_variable and tf.nn ops for graph mode.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

GLOROT_UNIFORM_INITIALIZER = tf.compat.v1.glorot_uniform_initializer()


def l2_regularizer(scale: float):
    def _fn(w):
        return scale * tf.nn.l2_loss(w)

    return _fn


def mlp_dense(
    x,
    units: int,
    scope: str,
    kernel_initializer=None,
    kernel_regularizer=None,
):
    """Fully connected: matmul + bias. Input last dim must be static."""
    if kernel_initializer is None:
        kernel_initializer = GLOROT_UNIFORM_INITIALIZER
    with tf.compat.v1.variable_scope(scope):
        in_dim = x.get_shape().as_list()[-1]
        if in_dim is None:
            raise ValueError(
                "mlp_dense needs a static feature dimension; got shape %s" % (x.get_shape(),)
            )
        in_dim = int(in_dim)
        w = tf.compat.v1.get_variable(
            "kernel",
            shape=[in_dim, units],
            initializer=kernel_initializer,
            regularizer=kernel_regularizer,
        )
        b = tf.compat.v1.get_variable(
            "bias",
            shape=[units],
            initializer=tf.compat.v1.zeros_initializer(),
        )
        return tf.nn.bias_add(tf.matmul(x, w), b)


def mlp_batch_norm(x, scope: str, epsilon: float = 1e-3):
    """Batch norm over batch (axis 0); uses batch mean/var (no moving averages)."""
    with tf.compat.v1.variable_scope(scope):
        n = x.get_shape().as_list()[-1]
        if n is None:
            raise ValueError("mlp_batch_norm needs static last dim; got %s" % (x.get_shape(),))
        n = int(n)
        beta = tf.compat.v1.get_variable("beta", [n], initializer=tf.compat.v1.zeros_initializer())
        gamma = tf.compat.v1.get_variable("gamma", [n], initializer=tf.compat.v1.ones_initializer())
        batch_mean, batch_var = tf.nn.moments(x, axes=[0], keepdims=False)
        return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)


def mlp_dropout(x, rate: float, name: Optional[str] = None):
    if rate and rate > 0:
        return tf.nn.dropout(x, rate=rate, name=name)
    return x
