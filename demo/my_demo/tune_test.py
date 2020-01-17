import os
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import keras
import tensorflow.keras.backend as K

from bertTAT.bert import get_custom_objects
from bertTAT.bert import load_trained_model_from_checkpoint
from bertTAT.bert import Tokenizer
from bertTAT.radam import RAdam


# 1. 提取预训练模型文件的路径
now_path = os.path.dirname(__file__)
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')


# 2. 参数设置
SEQ_LEN = 128
BATCH_SIZE = 2
EPOCHS = 1
LR = 1e-4


# 3. 使用tpu时的设置:TF_KERAS添加到环境变量，下面是使用TPU的设置
# from tensorflow.python import keras
# os.environ['TF_KERAS'] = '1'
# # Initialize TPU Strategy
# TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
# resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
# tf.contrib.distribute.initialize_tpu_system(resolver)
# strategy = tf.contrib.distribute.TPUStrategy(resolver)
# with strategy.scope():
#     model = load_trained_model_from_checkpoint(
#         config_path,
#         checkpoint_path,
#         training=True,
#         trainable=True,
#         seq_len=SEQ_LEN,
#     )


# 4. 读取token字典集，也可以往字典集添加token
token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 5. 定义分词器
tokenizer = Tokenizer(token_dict)


# 6. 加载数据并编码
def load_data(path):
    global tokenizer  # 全局变量
    indices, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), 'r') as reader:
                for line in reader.readlines():
                    ids, segments = tokenizer.encode(line, max_len=SEQ_LEN)
                    indices.append(ids)
                    sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return [indices, np.zeros_like(indices)], np.array(sentiments)


# 7. 定义训练集合测试集
train_x, train_y = load_data("./data")
test_x, test_y = load_data("./data")
# print("输出结果:", train_x, "\n", train_y)

# 8. 新建模型
model = load_trained_model_from_checkpoint(
        config_path,
        checkpoint_path,
        training=True,
        trainable=True,
        seq_len=SEQ_LEN,
)

inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=2, activation='softmax')(dense)
model = keras.models.Model(inputs, outputs)
model.compile(
    RAdam(lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)


# 9. 初始化变量
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)


# 10. 模型开始训练
model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# 11. 模型预测
predicts = model.predict(test_x, verbose=True).argmax(axis=-1)


# 12. 准确率打印
print(np.sum(test_y == predicts) / test_y.shape[0])
