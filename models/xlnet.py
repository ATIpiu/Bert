import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import XLNetModel, XLNetTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'xlnet'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 64  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.xlnet_path = './xlnet_pretrain'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.hidden_size = 768
        self.text = ""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.net = XLNetModel.from_pretrained(config.xlnet_path)
        for name, param in self.net.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.MLP = nn.Sequential(
            nn.Linear(768, 10, bias=True),
        ).to(config.device)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = x.long()
        x = self.net(x, output_all_encoded_layers=False).last_hidden_state
        x = F.dropout(x, self.alpha, training=self.training)
        x = torch.max(x, dim=1)[0]
        x = self.MLP(x)
        return torch.sigmoid(x)
