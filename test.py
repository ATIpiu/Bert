import time
import torch
import numpy as np
from train_eval import train, init_network,test,evaluate
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
if __name__ == '__main__':
    dataset = 'MyNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.test_path = dataset + '/data/test_data1.txt'
    train_data, dev_data, test_data = build_dataset(config)
    test_iter = build_iterator(test_data, config)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion ,predict,label= evaluate(config, model, test_iter, test=True)
    file=open("OutPut.txt","w+",encoding='utf-8')
    file1=open("MyNews/data/test_data1.txt")
    for line,pre in zip(file1,predict) :
        file.write(line[0:-2]+','+str(pre)+'\n');

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
