import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import *
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import convert_example, create_dataloader



# 语义表示
# model = AutoModel.from_pretrained('ernie-3.0-medium-zh')

MODEL_NAME = "ernie-1.0"

tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)

# print(model.pooler)
# print("out_features:",model.pooler.dense.weight.shape[0]) #获取预训练模型最后一层的输出特征数

class ErnieWithFC(nn.Layer):
    def __init__(self, ernie_model, fc_size, num_classes, dropout_prob=0.1):
        super(ErnieWithFC, self).__init__()
        self.ernie = ernie_model
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(fc_size, num_classes)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        _, pooled_output = self.ernie(input_ids, token_type_ids, position_ids)
        pooled_output = self.dropout(pooled_output)  # 应用dropout层
        output = self.fc(pooled_output)
        return output

    def save_pretrained(self,model_path):

        # assert not os.path.isfile(model_path), "Saving directory ({}) should be a directory, not a file".format(model_path)
        os.makedirs(model_path, exist_ok=True)

        paddle.save(self.state_dict(), model_path+"/ErnieWithFC.pdparams")

model = ErnieWithFC(ernie_model,ernie_model.pooler.dense.weight.shape[0],2)


# res = ernie_model(single_seg_input['input_ids'],single_seg_input['token_type_ids'])
# print("ernie_model res : ",res)

train_ds, dev_ds, test_ds = load_dataset(
    "chnsenticorp", splits=["train", "dev", "test"])

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


# for batch in train_data_loader:
#     print("batch 0 :",batch)
#     break
'''
一个batch堆叠出来的数据如下图所示
batch 0 : [Tensor(shape=[32, 128], dtype=int32, place=Place(gpu_pinned), stop_gradient=True,
       [[1   , 328 , 188 , ..., 0   , 0   , 0   ],
        [1   , 75  , 47  , ..., 0   , 0   , 0   ],
        [1   , 335 , 15  , ..., 0   , 0   , 0   ],
        ...,
        [1   , 47  , 10  , ..., 4   , 1172, 2   ],
        [1   , 317 , 42  , ..., 2   , 0   , 0   ],
        [1   , 185 , 45  , ..., 21  , 4   , 2   ]]), Tensor(shape=[32, 128], dtype=int32, place=Place(gpu_pinned), stop_gradient=True,
       [[0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        ...,
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0],
        [0, 0, 0, ..., 0, 0, 0]]), Tensor(shape=[32, 1], dtype=int64, place=Place(gpu_pinned), stop_gradient=True,
       [[1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1],
        [0],
        [0],
        [1],
        [1],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1]])]
'''


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

# global_step = 0
# for epoch in range(1, epochs + 1):
#     for step, batch in enumerate(train_data_loader, start=1):
#         input_ids, segment_ids, labels = batch
#         logits = model(input_ids, segment_ids)
#         loss = criterion(logits, labels)
#         probs = F.softmax(logits, axis=1)
#         correct = metric.compute(probs, labels)
#         metric.update(correct)
#         acc = metric.accumulate()
#
#         global_step += 1
#         if global_step % 10 == 0 :
#             #每10步打印一次
#             print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.clear_grad()
#     evaluate(model, criterion, metric, dev_data_loader)

# 如果有已经训练好的保存了的模型，可以直接加载模型参数
param_path = "./checkpoint_art/ErnieWithFC.pdparams"
param_dict = paddle.load(param_path)
model.set_dict(param_dict)
evaluate(model, criterion, metric, dev_data_loader)
# model.save_pretrained('./checkpoint_art')
# tokenizer.save_pretrained('./checkpoint_art')

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
