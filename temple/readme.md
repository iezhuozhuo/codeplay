### preprocessing.py相关模板，可供参考规范代码
统一使用`ModelCorpus`类作为数据读入预处理的`Processor`.

可支持工作有：
- 构建预料字典`field`：`self.field`:保存不同语料不同的词典(包括padding,unk，不同语料词典可以共享视任务而定)，可以完成`str2id`以及`id2str`功能，==但是`tokenizer`需要谨慎处理==。
- 构建`features pipeline`：`build()`中包含了三个函数--`read_data()`、`build_examples()`、`build_vocab()`
- 创建`dataloader`：`cread_batch()`
- padding分两种，padding词以及padding字符，分别使用：`padding_seq`,`padding_char_seq`,`padding_char_len`。
- 加载函数：`load_data()`、`load_field()`

### train_utils.py相关模板，可供参考规范代码

统一包含三部分：Metric计算、trainer、evaluate。可以定制的部分在参考模板中使用`FIXME`标注。

- trainer、evaluate两个重点改读入的数据，以及收集`pred`和`label`的方式
- Metric计算部分根据不同任务计算。

### run_train.py相关模板，可供参考规范代码


可以定制的部分在参考模板中使用`FIXME`标注，可修改定制。

### ModelConfig相关模板

定制读入modelconfig以及model。

