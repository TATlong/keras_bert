import keras
from keras.utils import plot_model
from bertTAT.bert import get_base_dict, get_model, compile_model, gen_batch_inputs

# # 英文输入示例
# sentence_pairs = [
#     [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
#     [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
#     [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
# ]

# 1. 中文输入示例
sentence_pairs_zh = ["我始终相信一句话#那就是天道酬勤!",
                   "今天必须把这个技术学习完#因为已经拖了很长时间了！",
                   "学好技术#早日实现梦想！",
                   "每天给自己设定一个目标#终究能人生巅峰。"
                   ]

sentence_pairs = []
for i in range(len(sentence_pairs_zh)):
    sentence = sentence_pairs_zh[i].split("#")
    sentence_pairs.append([])
    sentence_pairs[i].append(["".join(i) for i in sentence[0]])
    sentence_pairs[i].append(["".join(i) for i in sentence[1]])
print("sentence_pairs:", sentence_pairs)


# 2. 构建词典
token_dict = get_base_dict()
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())
print("token_dict:", token_dict)
print("token_list:", token_list)


# 3. 构建模型
model = get_model(
    token_num=len(token_dict),  # token 的数量
    head_num=5,  # 每个 transformer 中 multi-head attention 中 heads 的个数，默认12
    transformer_num=12,  # transformer的个数，默认12
    embed_dim=25,  # 嵌入维度，默认768
    feed_forward_dim=100,  # 每个 transformer 中 feed-forward 层的维度，默认3072
    seq_len=20,  # 输入序列的最大长度，为 None 时不限制。默认512
    pos_num=20,  # 最大 position 。默认512
    dropout_rate=0.05,  # dropout 的概率
)


# 4. 编译模型
compile_model(model)


# 5. 模型结构打印
# model.summary()


# 6. 保存网路结构为图片
plot_model(model,
           to_file=r"../model_pic/model_bert.png",
           show_shapes=True,
           show_layer_names=True)


def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,  # 列表，这个包含了许多 token 组成的句子对
            token_dict,  # 包括 BERT 所用的特殊符号在内的字典
            token_list,  # 包括所有 token 的列表
            seq_len=20,  # 序列的长度，默认512
            mask_rate=0.15,  # 随机 token 被替换为 [MASK] 的概率，然后预测这个被替换的 token。默认0.15
            swap_sentence_rate=0.5,  # 交换第一个句子和第二个句子的概率。默认0.5
        )


# 7. 模型训练
model.fit_generator(
    generator=_generator(),  # 训练集
    validation_data=_generator(),   # 测试集
    steps_per_epoch=10,
    epochs=1,
    validation_steps=100,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
)

# 8. 使用训练好的模型:取出 输入层 和 最后一个特征提取层
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,  # 如果为True，则将返回带有 MLM 和 NSP输出的模型；否则，将返回输入层和最后一个特征提取层。默认 True
    trainable=False,  # 模型是否是可训练的，默认和 training 一样的设置
    output_layer_num=4,  # 多少个FeedForward-Norm层的输出被连接为单个输出。仅在 training 为 False 时可用
)

"""关于training和trainable:
虽然看起来相似，但这两个参数是不相关的。
training表示是否在训练BERT语言模型，当为True时完整的BERT模型会被返回，当为False时没有MLM和NSP相关计算的结构，返回输入层和根据output_layer_num合并最后几层的输出。
加载的层是否可训练只跟trainable有关。

此外，trainable可以是一个包含字符串的列表，如果某一层的前缀出现在列表中，则当前层是可训练的。
在使用预训练模型时，如果不想再训练嵌入层，可以传入trainable=['Encoder']来只对编码层进行调整。
"""
