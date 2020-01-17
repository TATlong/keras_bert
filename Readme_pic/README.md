### 模型文件
该文件夹下是解压后的模型文件，当然需要提前下载好模型文件。<br>
主要包括以下的几个文件：<br>
![](https://github.com/TATlong/keras_bert/blob/master/pretrained_model/%E6%A8%A1%E5%9E%8B%E6%96%87%E4%BB%B6.png)

当然也可以通过代码下载模型到指定路径下：

    from keras_bert.datasets import get_pretrained, PretrainedList
    model_path = get_pretrained(PretrainedList.chinese_base)
