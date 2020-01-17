# -*- coding: UTF-8 -*-
from bertTAT.transformer_utils.feedforward import feed_forward_builder
from bertTAT.transformer_utils.multi_head_attention import attention_builder
from bertTAT.transformer_utils.wraper import _wrap_layer


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate
    :param trainable: Whether the layers are trainable
    :param use_adapter: Whether to use feed-forward adapters before each residual connections
    :param adapter_units: The dimension of the first transformation in feed-forward adapter
    :param adapter_activation: The activation after the first transformation in feed-forward adapter
    :return: Output layer
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name

    # 操作：Multi_Head_attention + Add + Norm
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )

    # 操作：Feed forward，两个连续的线性变换，这两个变换中间是一个激活函数，FFN(x)=max(0,X*W_1+b_1)*W_2+b_2
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )
    return feed_forward_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    last_layer = input_layer

    # 多个encoder的前向计算
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name='Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
            use_adapter=use_adapter,
            adapter_units=adapter_units,
            adapter_activation=adapter_activation,
        )
    return last_layer
