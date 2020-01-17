# -*- coding: UTF-8 -*-
from keras.layers import Layer
import keras.backend as K


class TriglePositiomEmbedding(Layer):
    """Position embedding use sine and cosine functions.
    See: https://arxiv.org/pdf/1706.03762

    Expand mode:
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self, mode=MODE_ADD, output_dim=None, **kwargs):
        if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
            if output_dim is None:
                raise NotImplementedError('`output_dim` is required in `%s` mode' % mode)
            if output_dim % 2 != 0:
                raise NotImplementedError('It does not make sense to use an odd output dimension: %d' % output_dim)
        self.mode = mode
        self.output_dim = output_dim
        self.supports_masking = True
        super(TriglePositiomEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'mode': self.mode,
            'output_dim': self.output_dim,
        }
        base_config = super(TriglePositiomEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
            pos_input = K.tile(K.expand_dims(K.arange(seq_len), axis=0), [batch_size, 1])
        elif self.mode == self.MODE_CONCAT:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
            pos_input = K.tile(K.expand_dims(K.arange(seq_len), axis=0), [batch_size, 1])
        else:
            output_dim = self.output_dim
            pos_input = inputs
        # 转换为32位浮点数
        if K.dtype(pos_input) != K.floatx():
            pos_input = K.cast(pos_input, K.floatx())

        evens = K.arange(output_dim // 2) * 2
        odds = K.arange(output_dim // 2) * 2 + 1

        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
        if self.mode == self.MODE_CONCAT:
            output = K.concatenate([inputs, output], axis=-1)
        if self.mode == self.MODE_ADD:
            output += inputs
        return output

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape


# if __name__ == '__main__':
#     from transformer_utils.embedding import EmbeddingRet, EmbeddingSim
#     import numpy as np
#     embed = EmbeddingRet(
#         input_dim=20,  # 输入的维度，即词典的大小
#         output_dim=10,  # 输出的维度，即embedding后的维度大小
#         mask_zero=True,  # 确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值
#         weights=[np.random.random((20, 10))],
#         trainable=True,
#         name='Token-Embedding',
#     )
#     p = K.constant([[i for i in range(9)]])
#     print("constant:", K.eval(p), p)
#     q = TriglePositiomEmbedding(
#         mode=TriglePositiomEmbedding.MODE_ADD,
#         name='Encoder-Embedding',
#     )(embed(p)[0])
#     print("input:", embed(p)[0].shape)
#     print("output:", q.shape)

