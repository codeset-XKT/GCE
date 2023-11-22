"""GCE方法概括

   一、初始化种群
    1. 顺序贪心策略：采用贪心法顺序搜索得到pred_len个个体
    2. 对错相关策略：做对全mask或做错全mask得到2个个体
    3. 技能相关策略：与该题相关的所有历史问题全mask或无关的全mask得到2个个体
    4. 复制这些个体 形成长度为pop_num的个体
   二、变异
    1. 新增取反变异操作，迭代次数越大取反变异概率越大
    2. 循环偏移算子
   三、交叉
    1. 精英集交叉
   四、自适应变异与交叉策略
    1. 交叉概率根据普适度和种群迭代次数来确定，普适度低交叉概率高，迭代次数低交叉概率高
    2. 对于普通变异操作，变异概率根据基因位以及迭代次数来确定，基因位越靠后变异概率高，迭代次数高变异率高（遗忘因素，越靠后的交互对重要性越大）
    3. 对于全变异操作，当种群普适度趋于稳定，适当增大全变异算子概率
    4. 外存经验矩阵（平滑因子and可信度）
    5. 正则化惩罚项，惩罚mask较多个体的普适度
"""

import random
import re
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import time

class GeneticAlgorithm_plus():
    def __init__(self, model, mask='0', pred_len=14, pop_num=None, iter_num=None, c=0.05, m=0.05, skill_num=123,
                 question_num=15678, matrix=True):
        self.model = model  # 预测模型对象
        self.mask = mask  # mask方式 '0'为0向量 'r'为去除
        self.pred_len = pred_len  # 预测长度 shape -> [1]
        self.pop_num = pop_num  # 种群数量
        self.iter_num = iter_num  # 迭代次数
        self.c = c  # 交叉概率
        self.m = m  # 变异概率
        self.cg = 0  # 当前进化代数
        self.elite_set = []  # 精英集，用于交叉
        self.a_v = 0  # 种群平均普适度
        self.best_v = 0  # 种群最高普适度
        self.best_fitness = []  # 截止当前代最好的个体
        self.best_x = None  # 种群最高普适度对应的mask序列
        self.skill_num = skill_num  # 知识点总个数
        self.question_num = question_num  # 问题总个数
        self.matrix = matrix  # 是否启用经验矩阵
        self.T1 = 1.2  # 相关度增率指数
        self.T2 = 1.0  # 置信度增率指数

        if matrix == True:
            self.update_num = 0  # 经验矩阵被更新的次数
            self.e_ICE = []  # 经验矩阵,个体因果效应
            self.e_ACE = []  # 经验矩阵,平均因果效应
            self.e_weight = []  # 经验矩阵,边更新次数

            arr = [0 for i in range(skill_num)]
            for i in range(skill_num):
                self.e_ICE.append(arr[:])
                self.e_ACE.append(arr[:])
                self.e_weight.append(arr[:])

        if self.pop_num is None:
            self.pop_num = 200 if self.pred_len.item() * 3 > 200 else self.pred_len.item() * 3
        if self.iter_num is None:
            self.iter_num = 200 if self.pred_len.item() * 3 > 200 else self.pred_len.item() * 3

    # 重置
    def reset(self):
        self.cg = 0  # 当前进化代数
        self.elite_set = []  # 精英集，用于交叉
        self.a_v = 0  # 种群平均普适度
        self.best_v = 0  # 种群最高普适度
        self.best_x = None  # 种群最高普适度对应的mask序列
        self.best_fitness = []  # 每一代最好的个体

    # 初始化
    def init(self, base):
        """初始化种群
            1. 采用贪心法顺序搜索得到pred_len个个体
            2. 做对全mask或做错全mask得到2个个体
            3. 与该题相关的所有历史问题全mask或无关的全mask得到2个个体
            4. 复制这些个体 形成长度为pop_num的个体
        """
        pop_list = []

        # 用贪心法顺序搜索得到pred_len个个体
        mask = np.array([0 for i in range(self.pred_len)], dtype=float)
        for i in range(len(mask)):
            item1 = torch.tensor(mask).to("cuda") * self.pad_data
            item1 = torch.where(torch.isnan(item1), torch.full_like(item1, self.question_num), item1).to(
                torch.long)  # mask值为零向量
            pad_predict1 = self.model(item1, self.pad_answer, self.pad_index)
            pack_predict1 = pack_padded_sequence(pad_predict1, self.pred_len.to("cpu"), enforce_sorted=True,
                                                 batch_first=True)
            y_pred1 = pack_predict1.data.cpu().contiguous().view(-1).detach()
            pred_list1 = []
            pred_list1.append(y_pred1)
            all_pred1 = torch.cat(pred_list1, 0)
            pred_pro1 = all_pred1.tolist()

            mask[i] = 1
            item2 = torch.tensor(mask).to("cuda") * self.pad_data
            item2 = torch.where(torch.isnan(item2), torch.full_like(item2, self.question_num), item2).to(
                torch.long)  # mask值为零向量
            pad_predict2 = self.model(item2, self.pad_answer, self.pad_index)
            pack_predict2 = pack_padded_sequence(pad_predict2, self.pred_len.to("cpu"), enforce_sorted=True,
                                                 batch_first=True)
            y_pred2 = pack_predict2.data.cpu().contiguous().view(-1).detach()
            pred_list2 = []
            pred_list2.append(y_pred2)
            all_pred2 = torch.cat(pred_list2, 0)
            pred_pro2 = all_pred2.tolist()

            if abs(base - pred_pro1[-1]) < abs(base - pred_pro2[-1]):
                pop_list.append(mask.copy())

        # 做对全mask或做错全mask得到2*2个个体
        answer_list = self.pad_answer.view(-1).tolist()

        pop_list.append(np.array(answer_list))
        pop_list.append(np.array(list(map(lambda x: (x + 1) % 2, answer_list))))

        # 与该题相关的所有历史问题全mask或无关的全mask得到2*2个个体
        mask = np.array([0 for i in range(self.pred_len)], dtype=float)
        ques_list = self.pad_data.view(-1).tolist()
        his_skills_list = []
        for i in range(len(ques_list)):
            his_skills_list.append(self.model.dict[str(ques_list[i])])
        for idx, his_skills in enumerate(his_skills_list):
            for his_skill in his_skills.split(','):
                for cur_skill in self.skills.split(','):
                    if cur_skill == his_skill:
                        mask[idx] = 1


        pop_list.append(mask)
        pop_list.append(np.array(list(map(lambda x: (x + 1) % 2, mask))))

        pop_list = np.array(pop_list, dtype=float)

        # 复制这些个体 形成长度为pop_num的个体
        pop_list = np.repeat(pop_list, self.pop_num // len(pop_list), axis=0)

        for i in range(self.pop_num - len(pop_list)):
            # 从先验策略中补齐剩余个体
            # pop_list = np.append(pop_list,pop_list[i].reshape(1,-1),axis=0)
            # 从随机策略中补齐剩余个体
            pop_list = np.append(pop_list, np.random.randint(0, 2, (self.pred_len.item())).reshape(1, -1), axis=0)
        return pop_list

    # 初始化精英集
    def initEliteSet(self, fitness, X):
        idx = np.argsort(-fitness)
        X = X[idx]
        # 精英集初始化
        self.elite_set = X[:self.pop_num // 6].tolist()

    # 编码
    def decode_X(self, X: np.array):
        """对整个种群的基因解码，上面的decode是对某个染色体的某个变量进行解码"""
        return X

    # 适者生存,个体选择
    def select(self, X, fitness, base):
        """根据轮盘赌法选择优秀个体"""
        # 个体选择加惩罚项
        alpha = 0.1 * (abs(0.5 - base) + 0.5) / X.shape[1]
        p = fitness + np.sum(X, axis=-1) * alpha

        # 拉大分布间的差距 让普适度大的更大可能被选中
        p = np.power(p, 1)
        p = p / p.sum()

        # p = fitness / fitness.sum()  # 归一化

        idx = np.array(list(range(X.shape[0])))
        # 这里的原理是根据普适度有放回的采样 意思是普适度高的容易被多次采样
        X2_idx = np.random.choice(idx, size=X.shape[0], p=p)  # 根据概率选择
        X2 = X[X2_idx, :]
        new_fitness = fitness[X2_idx]
        # 得到选择后的平均普适度
        self.a_v = np.mean(new_fitness)
        return X2, new_fitness

    # 自适应交叉策略,返回个体交叉概率值
    def crossover_pro(self, i_f):
        """i_f为个体普适度
           a_v为种群平均普适度
        """
        p_min = self.c
        if self.cg <= self.iter_num / 4:
            p_max = 0.07
        elif self.cg <= self.iter_num * 3 / 4:
            p_max = 0.06
        elif self.cg <= self.iter_num:
            p_max = 0.05

        if i_f >= self.a_v:
            p_c = p_max - (p_max - p_min) * (
                        self.cg / (2 * self.iter_num) + (i_f - self.a_v) / (2 * (self.best_v - self.a_v)))
        else:
            p_c = p_max
        return p_c

    # 产生交叉，这里是单个对应位交叉
    def crossover(self, X, fitness):
        """从精英集中选择优秀个体以概率c进行交叉操作"""
        for i in range(X.shape[0]):
            xa = X[i, :]
            xb = self.elite_set[random.randint(0, len(self.elite_set) - 1)][:]
            for j in range(X.shape[1]):
                # 产生0-1区间的均匀分布随机数，判断是否需要进行交叉替换
                if np.random.rand() <= self.crossover_pro(fitness[i]):
                    xa[j] = xb[j]
            X[i, :] = xa
        return X
        # """按顺序选择2个个体以概率c进行交叉操作"""
        # for i in range(0, X.shape[0], 2):
        #     xa = X[i, :]
        #     xb = X[i + 1, :]
        #     for j in range(X.shape[1]):
        #         # 产生0-1区间的均匀分布随机数，判断是否需要进行交叉替换
        #         if np.random.rand() <= self.c:
        #             xa[j], xb[j] = xb[j], xa[j]
        #     X[i, :] = xa
        #     X[i + 1, :] = xb
        # return X

    # 变异
    def mutation(self, X):
        flag = True
        if self.matrix:
            # 得到矩阵非零均值
            e_ACE = np.array(self.e_ACE)
            exist = (e_ACE != 0)
            if exist.sum() != 0:
                mean_ACE = e_ACE.sum() / exist.sum()
            else:
                mean_ACE = 0
            if np.max(e_ACE) - np.min(e_ACE) != 0:
                normal_ACE = (e_ACE - np.min(e_ACE)) / (np.max(e_ACE) - np.min(e_ACE))  # (0,1)
            else:
                normal_ACE = e_ACE

            # print("min",np.min(normal_ACE))
            # if np.mean(normal_ACE) == 0:
            #     mean_ACE = 0.5
            # else:
            #     mean_ACE = np.mean(normal_ACE)

        # 全变异操作，迭代次数越大变异概率越大
        all_m_p = 0
        if self.cg / self.iter_num >= 0.3:
            if np.mean(self.best_fitness[-self.pop_num // 3:]) == self.best_fitness[-1]:
                all_m_p = 0.03
            elif np.mean(self.best_fitness[-self.pop_num // 4:]) == self.best_fitness[-1]:
                all_m_p = 0.02
            elif np.mean(self.best_fitness[-self.pop_num // 5:]) == self.best_fitness[-1]:
                all_m_p = 0.01

        for i in range(X.shape[0]):
            if np.random.rand() <= all_m_p:
                X[i, :] = (X[i, :] + 1) % 2

            else:
                for j in range(X.shape[1]):
                    # 基础变异操作,迭代次数越大变异概率越小
                    if self.matrix == True:
                        # 融入经验矩阵
                        cur_skill = self.his_skill_list[j]
                        relevance = self.cal_relevance(cur_skill, self.skills, normal_ACE, mean_ACE)
                        # 各相关系数权重
                        c = 0.5
                        a = (1 - c) / 2
                        b = (1 - c) / 2
                        # 可信度
                        confidence = np.power(self.update_num / (self.skill_num ** 2),
                                              self.T2) if self.update_num < self.skill_num ** 2 else 1
                        # 平衡因子
                        balance = (1 - c * confidence) / (2 * a)
                        if np.random.rand() <= self.m * (
                                balance * a * np.log((self.cg / self.iter_num) + 1) + balance * b * np.log((j / X.shape[1]) + 1) + confidence * c * relevance):
                            # if np.random.rand() <= 0.01 + self.m*(self.cg/self.iter_num + j/X.shape[1])/2 + c*confidence*relevance:
                            X[i, j] = (X[i, j] + 1) % 2
                            # 对比循环偏移算子
                            # idx = np.random.randint(0, X.shape[1])
                            # a = X[i, j]
                            # X[i, j] = X[i, idx]
                            # X[i, idx] = a
                        if self.cg % 10 == 0 and flag:
                            flag = False
                            # print("confidence",confidence)
                            # print("relevance",relevance)
                    else:
                        if np.random.rand() <= self.m * (self.cg / self.iter_num + j / X.shape[1]) / 2:
                            X[i, j] = (X[i, j] + 1) % 2
        return X

    # 计算最佳个体的ACC
    def cal_AUC(self, mask, base):
        mask[mask == 0] = np.nan
        item = torch.tensor(mask).to("cuda") * self.pad_data
        item = torch.where(torch.isnan(item), torch.full_like(item, self.question_num), item).to(
            torch.long)  # mask值为去除
        item = item[0].cpu()[~np.isin(item[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to("cuda")
        answer = torch.tensor(mask).to("cuda") * self.pad_answer
        answer = torch.where(torch.isnan(answer), torch.full_like(answer, self.question_num), answer).to(
            torch.long)  # mask值为去除
        answer = answer[0].cpu()[~np.isin(answer[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to(
            "cuda")
        index = torch.concat((item[:, 1:], self.pad_index[:, -1:]), dim=-1)

        if item.size(-1) == 0:
            return [base, base, 0]

        pad_predict = self.model(item, answer, index)
        pack_predict = pack_padded_sequence(pad_predict, self.pred_len.to("cpu"), enforce_sorted=True,
                                            batch_first=True)
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
        pred_list = []
        pred_list.append(y_pred)
        all_pred = torch.cat(pred_list, 0)
        pred_pro = all_pred.tolist()
        # print('基础值', base, 'masked输出', pred_pro[-1])
        if base <= 0.5 and pred_pro[-1] <= 0.5:
            return [base, pred_pro[-1], 1]
        if base > 0.5 and pred_pro[-1] > 0.5:
            return [base, pred_pro[-1], 1]
        return [base, pred_pro[-1], 0]

    # 计算最佳个体的ODDS
    def cal_ODDS(self, ind, base):
        ind[ind == 0] = np.nan
        item = torch.tensor(ind).to("cuda") * self.pad_data
        item = torch.where(torch.isnan(item), torch.full_like(item, self.question_num), item).to(
            torch.long)  # mask值为去除
        item = item[0].cpu()[~np.isin(item[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to("cuda")
        answer = torch.tensor(ind).to("cuda") * self.pad_answer
        answer = torch.where(torch.isnan(answer), torch.full_like(answer, self.question_num), answer).to(
            torch.long)  # mask值为去除
        answer = answer[0].cpu()[~np.isin(answer[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to(
            "cuda")
        index = torch.concat((item[:, 1:], self.pad_index[:, -1:]), dim=-1)

        if item.size(-1) == 0:
            return 0

        pad_predict = self.model(item, answer, index)
        pack_predict = pack_padded_sequence(pad_predict, self.pred_len.to("cpu"), enforce_sorted=True,
                                            batch_first=True)
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
        pred_list = []
        pred_list.append(y_pred)
        all_pred = torch.cat(pred_list, 0)
        pred_pro = all_pred.tolist()
        odds_base = np.log(base/(1 - base))
        odds_ind = np.log(pred_pro[-1]/(1 - pred_pro[-1]))
        delta_odds = [odds_base, odds_ind]

        return delta_odds

    # 计算种群未被mask的初始值
    def predict_b(self):
        pad_predict = self.model(self.pad_data, self.pad_answer, self.pad_index)
        pack_predict = pack_padded_sequence(pad_predict, self.pred_len.to("cpu"), enforce_sorted=True, batch_first=True)
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
        pred_list = []
        pred_list.append(y_pred)
        all_pred = torch.cat(pred_list, 0)
        base = all_pred.tolist()[-1]
        return base

    # 计算种群普适度 mask为0向量
    def predict_0(self, X, base):
        pro_list = []
        for mask in X.copy():
            mask[mask == 0] = np.nan
            item = torch.tensor(mask).to("cuda") * self.pad_data
            item = torch.where(torch.isnan(item), torch.full_like(item, self.question_num), item).to(
                torch.long)  # mask值为零向量
            pad_predict = self.model(item, self.pad_answer, self.pad_index)
            pack_predict = pack_padded_sequence(pad_predict, self.pred_len.to("cpu"), enforce_sorted=True,
                                                batch_first=True)
            y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
            pred_list = []
            pred_list.append(y_pred)
            all_pred = torch.cat(pred_list, 0)
            pred_pro = all_pred.tolist()
            pro_list.append(pred_pro)

        effect_list = np.array(list(map(lambda x: abs(base - x[-1]), pro_list)))
        return effect_list

    # 计算种群普适度 mask为去除
    def predict_r(self, X, base):
        pro_list = []
        for mask in X.copy():
            mask[mask == 0] = np.nan
            item = torch.tensor(mask).to("cuda") * self.pad_data
            item = torch.where(torch.isnan(item), torch.full_like(item, self.question_num), item).to(
                torch.long)  # mask值为去除
            item = item[0].cpu()[~np.isin(item[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to("cuda")
            answer = torch.tensor(mask).to("cuda") * self.pad_answer
            answer = torch.where(torch.isnan(answer), torch.full_like(answer, self.question_num), answer).to(
                torch.long)  # mask值为去除
            answer = answer[0].cpu()[~np.isin(answer[0].cpu(), torch.tensor([self.question_num]))].view(1, -1).to(
                "cuda")
            index = torch.concat((item[:, 1:], self.pad_index[:, -1:]), dim=-1)

            if item.size(-1) == 0:
                continue

            pad_predict = self.model(item, answer, index)
            pack_predict = pack_padded_sequence(pad_predict, self.pred_len.to("cpu"), enforce_sorted=True,
                                                batch_first=True)
            y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()
            pred_list = []
            pred_list.append(y_pred)
            all_pred = torch.cat(pred_list, 0)
            pred_pro = all_pred.tolist()
            pro_list.append(pred_pro)

        effect_list = np.array(list(map(lambda x: abs(base - x[-1]), pro_list)))

        # 均值补全 去除全部题目所造成的预测输入缺失问题
        if len(effect_list) < self.pop_num:
            for i in range(self.pop_num - len(effect_list)):
                effect_list = np.append(effect_list, [sum(effect_list) / len(effect_list)])
        return effect_list

    # 更新经验矩阵
    def update_matrix(self, base, fitness, mask_list):
        self.update_num = self.update_num + 1
        idx = np.argwhere(mask_list == 0).reshape(-1)
        if len(idx) == 0:
            return
        his_skill_list = self.his_skill_list[idx]
        his_skill_list = ','.join(str(i) for i in his_skill_list)
        for skill in str(self.skills).split(','):
            for his_skill in his_skill_list.split(','):
                # 技能间ICE更新
                try:
                    self.e_ICE[int(his_skill)][int(skill)] += fitness
                except:
                    print(his_skill, skill)
                # 技能间采样次数
                self.e_weight[int(his_skill)][int(skill)] += 1
                # 技能间ACE更新
                self.e_ACE[int(his_skill)][int(skill)] = self.e_ICE[int(his_skill)][int(skill)] / \
                                                         self.e_weight[int(his_skill)][int(skill)]

    # 根据经验计算技能相关性
    def cal_relevance(self, skills1, skills2, normal_ACE, mean_ACE):
        relevance = 0
        count = 0
        for skill1 in str(skills1).split(','):
            for skill2 in str(skills2).split(','):
                i = int(skill1)
                j = int(skill2)
                if normal_ACE[i][j] == 0:
                    relevance += mean_ACE
                else:
                    relevance += normal_ACE[i][j]
                count += 1

        return np.power(relevance / count, self.T1)

    # 主要算法流程
    def run(self, pad_data, pad_answer, pad_index, skills, skill_list):
        start = time.perf_counter()

        self.pad_data = pad_data  # 题目序列 shape -> [1,pred_len]
        self.pad_answer = pad_answer  # 答案序列 shape -> [1,pred_len]
        self.pad_index = pad_index  # 下一时刻题目序列 shape -> [1,pred_len] 最后一道题对应的就是预测的题目
        self.skills = skills  # 预测题目的技能
        self.his_skill_list = np.array(skill_list)
        self.reset()

        # 根据mask方式选择对应mask的普适度函数
        if self.mask == '0':
            fitness_func = self.predict_0
        else:
            fitness_func = self.predict_r

        """遗传算法主函数"""
        # 未被mask的初始值
        base = self.predict_b()
        # 始化种群
        X0 = self.init(base)
        # 随机初始化种群，可以在随机初始化种群加约束
        # X0 = np.random.randint(0, 2, (self.pop_num, self.pred_len.item()))
        X0 = X0.astype(float)
        X1 = self.decode_X(X0)  # 染色体解码
        fitness = fitness_func(X1, base)  # 计算个体适应度
        # 初始化精英集
        self.initEliteSet(fitness, X1)
        # print("fitness.max()", fitness.max())
        self.best_v = fitness.max()
        self.best_x = X1[fitness.argmax()]
        self.best_fitness.append(fitness.max())

        g_best_fitness = [self.best_v]  # 每一代的最优普适度
        g_best_mask = [self.best_x]  # 每一代的最优序列
        for i in range(self.iter_num):
            self.cg = i + 1
            X2, new_fitness = self.select(X0, fitness, base)  # 选择操作
            X3 = self.crossover(X2, new_fitness)  # 交叉操作,理论上来说交叉完了会有一个新的普适度
            X4 = self.mutation(X3)  # 变异操作
            # 计算一轮迭代的效果
            X5 = self.decode_X(X4)
            fitness = fitness_func(X5, base)
            # print("fitness.max()", fitness.max())
            if fitness.max() > self.best_v:
                self.best_fitness.append(fitness.max())
                self.best_v = fitness.max()
                self.best_x = X5[fitness.argmax()]
                self.elite_set.pop(0)
                self.elite_set.append(self.best_x.tolist())  # 添加进精英集
            else:
                # self.best_fitness.append(self.best_v)
                self.best_fitness.append(fitness.max())
            X0 = X4
            # 记录每一代的最优普适度和个体
            g_best_fitness.append(fitness.max())
            g_best_mask.append(X5[fitness.argmax()])
            # 更新经验矩阵
            if self.matrix:
                self.update_matrix(base, g_best_fitness[-1], g_best_mask[-1])


        # 多次迭代后的最终效果
        print('-' * 50, 'plus ', self.matrix, '-' * 50)
        print("最大效应差是：", max(self.best_fitness))
        print("最优解是：", list(map(lambda x: np.nan if x == 0 else int(x), self.best_x)))

        end = time.perf_counter()
        print('运行时长', end - start)

        # 计算最佳个体的ACC
        best_masked_ind = (self.best_x + 1) % 2
        isRestore = self.cal_AUC(best_masked_ind, base)

        # 计算最佳个体的ODDS
        delta_odds = self.cal_ODDS(self.best_x.copy(), base)

        # print("self.best_fitness",self.best_fitness)
        return max(self.best_fitness), list(map(lambda x: np.nan if x == 0 else int(x), self.best_x)), self.best_fitness, isRestore, delta_odds