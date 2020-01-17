from bertTAT.bert import get_base_dict, get_model, compile_model, gen_batch_inputs

"""
从原始的输入句子对，构建要训练的样例
"""

# 英文输入示例
# sentence_pairs = [
#     [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
#     [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
#     [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
# ]


# 1. 中文输入示例句子对，即输入的原始句子对。
sentence_pairs_ = ["我始终相信一句话#那就是天道酬勤！",
                   "今天必须把这个技术学习完#因为已经拖了很长时间了！",
                   "学好技术#早日实现梦想！",
                   "每天给自己设定一个目标#终究能人生巅峰。",
                   "傻叉#来打我呀"
                   ]

# 2. 从输入的列表中构建句子对sentence_pairs
sentence_pairs = []
for i in range(len(sentence_pairs_)):
    sentence = sentence_pairs_[i].split("#")
    sentence_pairs.append([])
    sentence_pairs[i].append(["".join(i) for i in sentence[0]])
    sentence_pairs[i].append(["".join(i) for i in sentence[1]])
print("sentence_pairs:", sentence_pairs)


# 3. 构建自定义词典{token:id}，包含bert特殊的token加上自定义的token
token_dict = get_base_dict()
print("token_dict:", token_dict)
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)

# 4. 取出字典中所有的token
token_list = list(token_dict.keys())
print("token_list:", token_list)


# 5. 生成要送入模型的训练集（sample和label）
inputs, outputs = gen_batch_inputs(
    sentence_pairs,
    token_dict,
    token_list,
    seq_len=30,  # 句子的最大长度，长度不够的话补0，超过的话截断
    mask_rate=0.2,  # mask_rate:掩码的比例
    swap_sentence_rate=1.0,  # swap_sentence_rate:句子的swap阈值
)
print("inputs:", inputs)

print(".........................................")
print(".........................................")

# 6. 可视化训练数据

# inputs = [token_inputs, segment_inputs, masked_inputs]
# token_inputs：MASK+句子的swap
# segment_inputs：所在句子的下标（第一句or第二句）
# masked_inputs：MASK在token_inputs中的位置
token_dict = {value: key for key, value in token_dict.items()}
t1 = [t for t in inputs[0]]
for i in range(len(t1)):
    print("token_inputs:", [token_dict[i] for i in t1[i]])
    print("masked_inputs:", inputs[2][i])
    print("segment_inputs:", inputs[1][i])

print(".........................................")
print(".........................................")

# mlm（Maksed Language Model）outputs
for list_ in outputs[0]:
    p1 = []
    for j in list_:
        p1.append(token_dict[j[0]])
    print("out:", p1)

# NSP（Next Sentence Prediction）outputs
for i in outputs[1]:
    print("i:", i)
