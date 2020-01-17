# -*- coding: UTF-8 -*-
from bertTAT.transformer_utils.feedforward import FeedForward
from bertTAT.transformer_utils.layer_normalization import LayerNormalization
import keras


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True,
                use_adapter=False,
                adapter_units=None,
                adapter_activation='relu'):
    """Wrap layers with residual, normalization and dropout.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    build_output = build_func(input_layer)
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    # 是否使用adapter
    if use_adapter:
        adapter = FeedForward(
            units=adapter_units,
            activation=adapter_activation,
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
            name='%s-Adapter' % name,
        )(dropout_layer)
        dropout_layer = keras.layers.Add(name='%s-Adapter-Add' % name)([dropout_layer, adapter])

    # Add操作：dropout_layer+input_layer
    # dropout_layer：Multi-Head Attention后的向量（dropout）
    # input_layer：经过Embedding+Positional
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])

    # Norm操作：layer normalization，对每一个样本的feature进行normalization
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)

    return normal_layer
