from bertTAT.bert import load_vocabulary
from bertTAT.bert import load_trained_model_from_checkpoint
from bertTAT.bert import Tokenizer

import numpy as np
import os

"""
加载预训练模型，并输出token的embedding向量
"""


"""联网下载模型，并返回配置信息"""
# from bertTAT.bert import get_pretrained, PretrainedList, get_checkpoint_paths
# model_path = get_pretrained(PretrainedList.multi_cased_base)
# paths = get_checkpoint_paths(model_path)
# print(paths.config, paths.checkpoint, paths.vocab)


"""建议把模型下载到本地，并加载模型"""

# 测试样例
text = '语言模型'

now_path = os.path.dirname(__file__)

# 1. 提取预训练模型文件的路径
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# 2. 使用load_vocabulary() 函数, 传入vocab_path
token_dict = load_vocabulary(vocab_path)

# 3. 加载预训练模型
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

# 4. 实例化Tokenizer分词对象
tokenizer = Tokenizer(token_dict)

# 5. 开始分词
tokens = tokenizer.tokenize(text)
# ['[CLS]', '语', '言', '模', '型', '[SEP]']

# 6. 句子编码,返回token下标和segment下标
indices, segments = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
# [101, 6427, 6241, 3563, 1798, 102, 0, 0, 0, 0]
print(segments[:10])
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 7. 提取token的embedding特征
predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, ":=>", predicts[i].tolist()[:5])

