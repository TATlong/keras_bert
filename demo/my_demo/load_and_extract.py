import os
import codecs
import numpy as np
from bertTAT.bert import Tokenizer
from bertTAT.bert import load_trained_model_from_checkpoint


# 1. 提取预训练模型文件的路径
now_path = os.path.dirname(__file__)
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# 2. 添加字典
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 3. 建立模型
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
model.summary(line_length=120)


# 4. 分词
tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
print(segments[:10])


# 5. 提取特征
predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])
