import codecs, os
import numpy as np
from bertTAT.bert import extract_embeddings

now_path = os.path.dirname(__file__)
pretrained_path = now_path+"/../pretrained_model/chinese_L-12_H-768_A-12"

with codecs.open('ceshi.txt', 'r', 'utf8') as reader:
    texts = map(lambda x: x.strip(), reader)
    embeddings = extract_embeddings(pretrained_path, texts)
    print(np.array(embeddings[0]).shape)
