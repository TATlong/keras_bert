from bertTAT.bert import extract_embeddings
from bertTAT.bert import load_trained_model_from_checkpoint

import numpy as np
import os, codecs

# 1. 提取预训练模型文件的路径
now_path = os.path.dirname(__file__)
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"


# 1. 如果不需要微调，只想提取词/句子的特征，如提取每个句子对应的全部词的特征
texts = ["世上无难事", '只要肯攀登!']
embeddings = extract_embeddings(pretrained_path, texts)
print("embedding:", np.array(embeddings[0]).shape)


# 2. 输入是成对的句子，想使用最后4层特征，且提取NSP位位置输出和max-pooling的结果
# 输出结果中不再包含词的特征，NSP和max-pooling的输出会拼接在一起，每个numpy数组的大小为(768 x 4 x 2,)
from bertTAT.bert import extract_embeddings, POOL_NSP, POOL_MAX
texts = [
    ('公司加班很严重', '但也要保持学习！'),
    ('算法学习', '永不止步。')
]
embeddings = extract_embeddings(pretrained_path, texts, output_layer_num=4, poolings=[POOL_NSP, POOL_MAX])
print("句子对:", np.array(embeddings).shape)


# 3. 可以使用adapter来对预训练模型进行微调,下面的代码只让adapter和layer normalization成为可训练的层
layer_num = 12
config_path = os.path.join(pretrained_path, 'bert_config.json')
model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(
    config_path,
    model_path,
    training=False,
    use_adapter=True,
    trainable=
    ['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(layer_num)],
)


# 4. 使用任务嵌入:如果有多任务训练的需求，可以启用任务嵌入层,
# 针对不同任务将嵌入的结果加上不同的编码，注意要让Embedding-Task层可训练
from bertTAT.bert import load_trained_model_from_checkpoint
config_path = os.path.join(pretrained_path, 'bert_config.json')
model_path = os.path.join(pretrained_path, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(
    config_file=config_path,
    checkpoint_file=model_path,
    training=False,
    trainable=True,
    use_task_embed=True,
    task_num=10,
)