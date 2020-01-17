import os
import codecs
import numpy as np

import tensorflow as tf
from bertTAT.bert import get_custom_objects
from bertTAT.bert import Tokenizer
from bertTAT.bert import load_trained_model_from_checkpoint


# 1. 提取预训练模型文件的路径
now_path = os.path.dirname(__file__)
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# TPU使用的设置
TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
tf.contrib.distribute.initialize_tpu_system(resolver)
strategy = tf.contrib.distribute.TPUStrategy(resolver)


# 加载基础模型
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

with strategy.scope():
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)


# 提取特征
tokenizer = Tokenizer(token_dict)
text = 'From that day forth... my arm changed... and a voice echoed'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=512)
print(tokens)

predicts = model.predict([np.array([indices] * 8), np.array([segments] * 8)])[0]

for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:19])
