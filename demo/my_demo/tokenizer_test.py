from bertTAT.bert import Tokenizer

# 字典存放着 token 和 id 的映射,字典里还有BERT特有的 token；
# 文本拆分出来的字在字典不存在，它的 id 会是 5，代表 [UNK]，即 unknown
token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,
    'un': 2,
    '##aff': 3,
    '##able': 4,
    '[UNK]': 5,
    '明': 6,
    '天': 7,
}


# 分词：中文的话，在CJK字符集内的中文以单字分割；英文的话采用最大贪心匹配
tokenizer = Tokenizer(token_dict)
print(tokenizer.tokenize("上班使我快乐！", "明天还要上班！"))
print(tokenizer.tokenize("unaffable"))
print(tokenizer.tokenize('unaffable', '钢'))

# 拆分不存在字典中的单词，结果如下，可以看到英语中会直接把不存在字典中的部分直接按字母拆分
# ['[CLS]', 'un', '##k', '##n', '##o', '##w', '##n', '[SEP]']
print(tokenizer.tokenize('unknown'))


# 英文：下标和段落下标
strs = 'unaffable'
print("分词结果:", strs, tokenizer.tokenize(strs))
indices, segments = tokenizer.encode(strs)
print("对应token下标:{}".format(indices))
print("段落下标:", segments)


# 中文：下标和段落下标,同时可以指定截取的最大长度
print("分词结果:", tokenizer.tokenize("上班使我快乐！", "明天还要上班！"))
indices, segments = tokenizer.encode("上班使我快乐！", "明天还要上班！")
print("对应的token下标:", indices)
print("段落下标:", segments)


# max_len：最大的字下标长度，如果拆分完的字不超过max_len，则用 0 填充
print("分词结果:", tokenizer.tokenize(first='unaffable', second='钢'))
indices, segments = tokenizer.encode(first='unaffable', second='钢', max_len=10)
print("对应的token下标:", indices)
print("段落下标:", segments)


# 匹配分词后的结果在原始文本中的起始和终止下标，只是绝对位置的匹配，从运行结果更容易看清除
print("原始文本:", tokenizer.tokenize(first="All rights reserved."))
print(Tokenizer.rematch("All rights reserved.", ["[UNK]", "righs", "[UNK]", "ser", "[UNK]", "[UNK]"]))
print("原始文本:", tokenizer.tokenize(first="嘛呢，吃了吗？"))
print("对应的下标:", tokenizer.encode(first="嘛呢，吃了吗？"))
print("匹配结果:", Tokenizer.rematch("你嘛呢，吃了吗？", ["你", "呢", "[UNK]",  "，", "[UNK]", "了", "吗", "？"]))
