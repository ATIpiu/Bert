import time
import torch
import numpy as np
from train_eval import train, init_network, test, evaluate
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--test', type=str, required=False, help='choose a testType :single or file')
parser.add_argument('--data', type=str, required=True, help='choose a dataset :single or file')
args = parser.parse_args()


def singleTest(model, config, text):
    start_time = time.time()
    config.batch_size = 1
    config.text = text
    test_data = build_dataset(config, True)
    test_iter = build_iterator(test_data, config)
    model.eval()
    test_acc, test_loss, test_report, test_confusion, predict, label = evaluate(config, model, test_iter, test=True)
    print(predict[0])
    end_time = time.time()
    print("Time usage:", int(round(end_time * 1000 - start_time * 1000)))


def fileTest(model, config, filepath=""):
    if filepath != "":
        config.test_path = filepath
    test_data = build_dataset(config,True)
    start_time = time.time()
    test_iter = build_iterator(test_data, config)
    model.eval()
    test_acc, test_loss, test_report, test_confusion, predict, label = evaluate(config, model, test_iter, test=True)
    file = open(dataset + "/output/OutPut.txt", "w+", encoding='utf-8')
    file1 = open(dataset + "/data/test.txt", encoding='utf-8')
    for line, pre in zip(file1, predict):
        file.write(line[0:-1] + ',' + str(pre) + '\n');
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    dataset = args.data # 数据集
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    if args.test == 'text':
        singleTest(model, config, "《高殿战记》再次迎来一波小更新。此次更新内容丰富，制作组诚意拉满。让我们来一睹为快！开发者收到多位玩家反馈，更新了制作界面。物品等	@7")
    else:
        fileTest(model, config, config.test_path)

