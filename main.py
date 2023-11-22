"""对比实验 2022-1-23
"""

import torch
from GCE import GeneticAlgorithm_plus
from torch.nn.utils.rnn import pack_padded_sequence

from ModelFile.main import parse_args
from ModelFile.model import DKT
from ModelFile.model import SAKT

import csv
import time

def calLen(arr_list):
    str_list = ','.join(str(i) for i in arr_list).replace('nan', '0').split(',')
    int_list = list(map(lambda x:int(x),str_list))
    return len(int_list) - sum(int_list)

if __name__ == "__main__":

    # 指定模型和数据集以及预测长度 #
    model_name = 'DKT'
    dataset = 'ASSIST09'
    length = 15
    # 指定模型和数据集以及预测长度 #

    path_dict = {
        'DKT_ASSIST09':'./param/DKT_ASSIST09_QE_0.737.pkl',
        'SAKT_ASSIST09':'./param/SAKT_ASSIST09_QE_0.751.pkl',
        'DKT_EdNet': './param/DKT_EdNet_0.756-NEW.pkl',
        'SAKT_EdNet': './param/SAKT_EdNet_QE_0.757.pkl',
    }

    skill_num_dict = {
        'ASSIST09':123,
        'EdNet':188
    }

    model_list = ['DKT','SAKT']
    dataset_list = ['ASSIST09','EdNet']

    # 模型参数路径
    path = path_dict[model_name+'_'+dataset]

    # 复杂度
    T_dict = {
        14:40,
        15:50,
        25:80,
        26:80,
        30:100
    }

    # 模型配置项
    args = parse_args(dataset)
    if model_name == 'DKT':
        model = DKT(args).to(args.device)
    else:
        model = SAKT(args).to(args.device)

    model.load_state_dict(torch.load(path))
    model.eval()

    count = 0
    # 预测长度
    pred_len = torch.tensor([length]).to('cuda')
    # 技能总个数
    skill_num = skill_num_dict[dataset]

    # 获取的序列个数
    seq_num = 0

    # plus版全局初始化
    geneticAlgorithm_plus_matrix = GeneticAlgorithm_plus(
        model=model,
        mask='r',
        pred_len=pred_len,
        skill_num=skill_num,
        c=0.05,
        m=0.05,
        question_num=-1,
        iter_num=T_dict[length],
        pop_num=T_dict[length]
    )

    with open(r'.\data\{0}\train_test\test_question.txt'.format(dataset),'r') as f:
        line = f.read().split('\n')
        while seq_num <= 299:
            if len(line[count])==0 or int(line[count])<length+1:
                count = count + 3
                continue
            seq_num = seq_num + 1
            q = list(map(lambda x:int(x),line[count+1].split(',')))
            a = list(map(lambda x:int(x),line[count+2].split(',')))
            pad_data = torch.tensor(q[:length]).view(1,-1).to("cuda")
            pad_answer = torch.tensor(a[:length]).view(1,-1).to("cuda")
            pad_index = torch.tensor(q[1:length+1]).view(1,-1).to("cuda")
            pad_label = torch.tensor(a[1:length+1]).view(1,-1).to("cuda")
            seq_lens = torch.tensor([pad_index.size(1)]).to("cuda")

            # 无mask预测值
            pad_predict = model(pad_data, pad_answer, pad_index)
            pack_predict = pack_padded_sequence(pad_predict, seq_lens.to("cpu"), enforce_sorted=True,batch_first=True)
            y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
            pred_list = []
            pred_list.append(y_pred)
            all_pred = torch.cat(pred_list, 0)

            # 真实标签
            label = int(pad_label[0][-1])
            # 原预测值
            base = all_pred.tolist()[-1]
            # 历史预测题目
            ques_list = pad_data.tolist()[0]
            # 当前预测题目
            ques = pad_index[0][-1]
            # 历史预测技能序列
            skill_list = list(map(lambda x:model.dict[str(x)],pad_data[0].tolist()))
            # 当前预测技能
            skills = model.dict[str(pad_index[0,-1].item())]

            gp_max_m,gp_list_m,gp_iter_list_m, gp_isRestore, gp_odds = geneticAlgorithm_plus_matrix.run(
                pad_data=pad_data,
                pad_answer=pad_answer,
                pad_index=pad_index,
                skills=skills,
                skill_list=skill_list
            )

            count = count + 3
            print('\n\n')