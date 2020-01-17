## 简介
keras_bert 是 CyberZHG 大佬封装好了Keras版的Bert，可以直接调用官方发布的预训练权重。<br>

github：<https://github.com/CyberZHG/keras-bert>

这里只是简单的把一些大佬封装的依赖包整到了一起而已，主要是方便以后的研究和学习，详细的bert使用请参考：

<https://github.com/CyberZHG/keras-bert/blob/master/README.zh-CN.md>


## demo中的部分解释



## 一些简单的模型的训练和使用

get_model() 来取得 BERT 模型，它有以下参数可供选择 <br>
    
* token_num：token 的数量 <br>
* pos_num：最大 position 。默认512 <br>
* seq_len：输入序列的最大长度，为 None 时不限制。默认512 <br>
* embed_dim：嵌入维度，默认768
* transformer_num：transformer的个数，默认12
* head_num：每个 transformer 中 multi-head attention 中 heads 的个数，默认12
* feed_forward_dim：每个 transformer 中 feed-forward 层的维度，默认3072
* dropout_rate：dropout 的概率
* attention_activation：attention 层的激活函数
* feed_forward_activation：feed forward 层使用的激活函数，默认是gelu
* training：如果为True，则将返回带有 MLM 和 NSP输出的模型；否则，将返回输入层和最后一个特征提取层。默认 True
* trainable：模型是否是可训练的，默认和 training 一样的设置
* output_layer_num：多少个FeedForward-Norm层的输出被连接为单个输出。仅在 training 为 False 时可用。默认1
* use_task_embed：是否将 task embedding 加到现有的 embedding 中，默认 False
* task_num：任务数，默认10
* use_adapter：是否在每个残差网络前使用 feed-forward adapter，默认 False
* adapter_units：feed-forward adapter 中第一个 transformation 的维度 <br>
关于adapter可以参考这篇论文：<https://arxiv.org/pdf/1902.00751.pdf>

gen_batch_inputs() 函数可以产生我们用于训练的数据，可用参数如下:

* sentence_pairs：列表，这个包含了许多 token 组成的句子对。
* token_dict：包括 BERT 所用的特殊符号在内的字典
* token_list：包括所有 token 的列表
* seq_len：序列的长度，默认512
* mask_rate：随机 token 被替换为 [MASK] 的概率，然后预测这个被替换的 token。默认0.15
* mask_mask_rate：如果一个 token 要被替换为 [MASK]，真正替换为 [MASK] 的概率。默认0.8
* mask_random_rate：如果一个 token 要被替换为 [MASK]，替换成一个随机的 token。默认0.1
* swap_sentence_rate：交换第一个句子和第二个句子的概率。默认0.5
* force_mask：至少一个位置的 token 被 masked，默认 True <br>

compile_model() 函数用来编译我们的模型，可用参数如下:

* model：要编译的模型
* weight_decay：权重衰减率，默认0.01
* decay_steps：学习率会在这个步长中线性衰减至0，默认100000
* warmup_steps：学习率会在预热步长中线性增长到设置的学习率，默认10000
* learning_rate：学习率，默认1e-4 <br>

warmup可以参考这篇文章：<https://yinguobing.com/tensorflowzhong-de-xue-xi-lu-re-shen/>
当step小于warm up setp时，学习率等于基础学习率×(当前step/warmup_step)，由于后者是一个小于1的数值，因此在整个warm up的过程中，学习率是一个递增的过程！当warm up结束后，学习率开始递减。

load_trained_model_from_checkpoint() 函数用来加载官方训练好的模型，可用参数如下:

* config_file：JSON 配置文件路径
* checkpoint_file：checkpoint 文件路径
* training：True 的话，会返回整个模型，否则会忽略 MLM 和 NSP 部分。默认 False
* trainable：模型是否可训练，默认和 training 设置一样
* output_layer_num：多少个FeedForward-Norm层的输出被连接为单个输出。仅在 training 为 False 时可用。默认1
* seq_len：如果这个数值比配置文件中的长度小，position embeddings 会被切成适用于这个长度。默认1e9 <br>

## 一些链接

谷歌BERT地址：<https://github.com/google-research/bert> <br>
中文预训练BERT-wwm：<https://github.com/ymcui/Chinese-BERT-wwm>
