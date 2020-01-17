# -*- coding: UTF-8 -*-
from bertTAT.transformer_utils.layer_normalization import LayerNormalization
from bertTAT.transformer_utils.multi_head_attention import MultiHeadAttention, attention_builder
from bertTAT.transformer_utils.feedforward import FeedForward
from bertTAT.transformer_utils.triangle_position_embedding import TriglePositiomEmbedding
from bertTAT.transformer_utils.embedding import EmbeddingRet, EmbeddingSim
from bertTAT.transformer_utils.encoders import get_encoders, get_encoder_component
from bertTAT.transformer_utils.decoders import get_decoders, decode, get_decoder_component
from bertTAT.transformer_utils.feedforward import feed_forward_builder
from .backend import keras

__all__ = [
    'get_custom_objects', 'get_encoders', 'get_decoders', 'get_model', 'decode',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component', 'get_decoder_component',
]


def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TriglePositiomEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


def get_model(token_num,
              embed_dim,
              encoder_num,
              decoder_num,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation='relu',
              dropout_rate=0.0,
              use_same_embed=True,
              embed_weights=None,
              embed_trainable=None,
              trainable=True,
              use_adapter=False,
              adapter_units=None,
              adapter_activation='relu'):
    """Get full model without compilation.

    :param token_num: Number of distinct tokens.
    :param embed_dim: Dimension of token embedding.
    :param encoder_num: Number of encoder components.
    :param decoder_num: Number of decoder components.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param use_same_embed: Whether to use the same token embedding layer. `token_num`, `embed_weights` and
                           `embed_trainable` should be lists of two elements if it is False.
    :param embed_weights: Initial weights of token embedding.
    :param embed_trainable: Whether the token embedding is trainable. It will automatically set to False if the given
                            value is None when embedding weights has been provided.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Keras model.
    """
    if not isinstance(token_num, list):
        token_num = [token_num, token_num]

    encoder_token_num, decoder_token_num = token_num

    if not isinstance(embed_weights, list):
        embed_weights = [embed_weights, embed_weights]

    encoder_embed_weights, decoder_embed_weights = embed_weights    # encoder_embed_trainable转换为列表，例如：(13, 30)
    if encoder_embed_weights is not None:
        encoder_embed_weights = [encoder_embed_weights]             # 再转换成高维的，例如:(1，13, 30)
    if decoder_embed_weights is not None:
        decoder_embed_weights = [decoder_embed_weights]

    if not isinstance(embed_trainable, list):
        embed_trainable = [embed_trainable, embed_trainable]

    encoder_embed_trainable, decoder_embed_trainable = embed_trainable
    if encoder_embed_trainable is None:
        encoder_embed_trainable = encoder_embed_weights is None
    if decoder_embed_trainable is None:
        decoder_embed_trainable = decoder_embed_weights is None

    # 词或字嵌入层
    if use_same_embed:
        encoder_embed_layer = decoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Token-Embedding',
        )
    else:
        encoder_embed_layer = EmbeddingRet(
            input_dim=encoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=encoder_embed_weights,
            trainable=encoder_embed_trainable,
            name='Encoder-Token-Embedding',
        )
        decoder_embed_layer = EmbeddingRet(
            input_dim=decoder_token_num,
            output_dim=embed_dim,
            mask_zero=True,
            weights=decoder_embed_weights,
            trainable=decoder_embed_trainable,
            name='Decoder-Token-Embedding',
        )

    # Positional嵌入层
    encoder_input = keras.layers.Input(shape=(None,), name='Encoder-Input')
    encoder_embed = TriglePositiomEmbedding(
        mode=TriglePositiomEmbedding.MODE_ADD,
        name='Encoder-Embedding',
    )(encoder_embed_layer(encoder_input)[0])

    # encoder层
    encoded_layer = get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
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
    decoder_input = keras.layers.Input(shape=(None,), name='Decoder-Input')
    decoder_embed, decoder_embed_weights = decoder_embed_layer(decoder_input)
    decoder_embed = TriglePositiomEmbedding(
        mode=TriglePositiomEmbedding.MODE_ADD,
        name='Decoder-Embedding',
    )(decoder_embed)

    # decoder层
    decoded_layer = get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
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
    dense_layer = EmbeddingSim(
        trainable=trainable,
        name='Output',
    )([decoded_layer, decoder_embed_weights])

    return keras.models.Model(inputs=[encoder_input, decoder_input], outputs=dense_layer)


