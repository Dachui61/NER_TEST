import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
import paddle
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import convert_example, create_dataloader

train_ds, dev_ds, test_ds = load_dataset(
    "chnsenticorp", splits=["train", "dev", "test"])

print(train_ds)

print(train_ds.label_list)

for data in train_ds.data[:5]:
    print(data)

MODEL_NAME = "ernie-1.0"

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# 将原始输入文本切分token，
tokens = tokenizer._tokenize("请输入测试样例")
print("Tokens: {}".format(tokens))

# token映射为对应token id
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))


# 拼接上预训练模型对应的特殊token ，如[CLS]、[SEP]
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)

# 转化成paddle框架数据格式
tokens_pd = paddle.to_tensor([tokens_ids])
print("Tokens : {}".format(tokens_pd))

# 此时即可输入ERNIE模型中得到相应输出
sequence_output, pooled_output = ernie_model(tokens_pd)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))

print("=================================")

# 单句输入
single_seg_input = tokenizer(text="请输入测试样例")
# 句对输入
multi_seg_input = tokenizer(text="请输入测试样例1", text_pair="请输入测试样例2")

print("单句输入token (str): {}".format(tokenizer.convert_ids_to_tokens(single_seg_input['input_ids'])))
print("单句输入token (int): {}".format(single_seg_input['input_ids']))
print("单句输入segment ids : {}".format(single_seg_input['token_type_ids']))

print()
print("句对输入token (str): {}".format(tokenizer.convert_ids_to_tokens(multi_seg_input['input_ids'])))
print("句对输入token (int): {}".format(multi_seg_input['input_ids']))
print("句对输入segment ids : {}".format(multi_seg_input['token_type_ids']))

print("=================================")

# 模型运行批处理大小
batch_size = 32
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

# print("train_data_loader",train_data_loader)
# ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# In the code you provided, num_classes is used to specify the number of classes in the classification task.
# The ErnieForSequenceClassification model is a pre-trained transformer model that has been fine-tuned for sequence classification tasks. When we initialize this model, we need to specify the number of output classes that the model should predict. In other words, it is the number of labels or categories that our classification model will be trained to classify.
# For example, if we are working on a binary classification problem (e.g., sentiment analysis where the task is to classify whether a given text expresses a positive or negative sentiment), then we would set num_classes=2. If we are working on a multi-class classification problem (e.g., classifying news articles into different topics such as politics, sports, entertainment), then we would set num_classes to the number of classes in the dataset (i.e., the number of different topics).

model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=len(train_ds.label_list))

# 设置Fine-Tune优化策略，接入评价指标
from paddlenlp.transformers import LinearDecayWithWarmup

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 1 #3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

import paddle.nn.functional as F
from utils import evaluate

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            #每10步打印一次
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)

model.save_pretrained('./checkpoint')
tokenizer.save_pretrained('./checkpoint')

# 如果有已经训练好的保存了的模型，可以直接加载模型参数
# param_path = "./checkpoint/model_state.pdparams"
# param_dict = paddle.load(param_path)
# model.set_dict(param_dict)
# evaluate(model, criterion, metric, dev_data_loader)

from utils import predict

data = [
    {"text":'这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'},
    {"text":'怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'},
    {"text":'作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'},
]
label_map = {0: 'negative', 1: 'positive'}

results = predict(
    model, data, tokenizer, label_map, batch_size=batch_size)
for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text, results[idx]))
