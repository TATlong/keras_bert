# -*- coding: UTF-8 -*-
from bertTAT.transformer_utils.wraper import _wrap_layer
from bertTAT.transformer_utils.multi_head_attention import attention_builder
from bertTAT.transformer_utils.feedforward import feed_forward_builder
import numpy as np


def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat


def decode(model,
           tokens,
           start_token,
           end_token,
           pad_token,
           top_k=1,
           temperature=1.0,
           max_len=10000,
           max_repeat=10,
           max_repeat_block=10):
    """Decode with the given model and input tokens.

    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param top_k: Choose the last token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :param max_len: Maximum length of decoded list.
    :param max_repeat: Maximum number of repeating blocks.
    :param max_repeat_block: Maximum length of the repeating block.
    :return: Decoded tokens.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
        for i in range(len(predicts)):
            if top_k == 1:
                last_token = predicts[i][-1].argmax(axis=-1)
            else:
                probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
                probs.sort(reverse=True)
                probs = probs[:top_k]
                indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
                probs = np.array(probs) / temperature
                probs = probs - np.max(probs)
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                last_token = np.random.choice(indices, p=probs)
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or\
                    (max_len is not None and output_len >= max_len) or\
                    _get_max_suffix_repeat_times(decoder_inputs, max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]
    return outputs


def get_decoder_component(name,
                          input_layer,
                          encoded_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention, multi-head query attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
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
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name

    # Masked Multi-head self-attention + Add + Norm
    self_attention_layer = _wrap_layer(
        name=self_attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=self_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,      # 只使用历史时刻的信息做attention
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )

    # Multi-head Query Attention + Add + Norm
    query_attention_layer = _wrap_layer(
        name=query_attention_name,
        # Q：self_attention_layer， K：encoded_layer， V：encoded_layer，encoded_layer是固定的
        input_layer=[self_attention_layer, encoded_layer, encoded_layer],
        build_func=attention_builder(
            name=query_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
    )

    # Feed forward + Add + Norm
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=query_attention_layer,
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


def get_decoders(decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
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

    # 多个decoder组件的前向计算
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name='Decoder-%d' % (i + 1),
            input_layer=last_layer,
            encoded_layer=encoded_layer,
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
