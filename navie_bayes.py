# -*- coding: UTF-8 -*-
import time
import random

# 常量定名（不能更改）
AGE = 0
WORKCLASS = 1
FNLWGT = 2
EDUCATION= 3
EDUCATION_NUM = 4
MARITAL_STATUS = 5
OCCUPATION = 6
RELATIONSHIP = 7
RACE = 8
SEX = 9
CAPITAL_GAIN = 10
CAPITAL_LOSS = 11
HOURS = 12
COUNTRY = 13
CATEGORY = 14


def read_file(filename: str, split: str) -> []:
    """ 读取文件
    :param filename: 文件名
    :param split: 分隔符
    :return: lines: []

    lines[] 数组格式
    ["值1", "值2, ...] 第一行
    ["值1", "值2, ...] 第二行
    ["值1", "值2, ...] 第三行
    ... 第n行
    """
    lines = []
    with open(filename, 'r') as f:  # 打开文件
        file_content = f.readlines()  # 读取文件
        for line in file_content:  # 遍历行
            if line == '\n':  # 排除空行
                continue
            if line == '':
                continue
            # 去除换行符，用分隔符切割元素
            fields = line.strip().split(split)
            for i in range(len(fields)):
                # 去除多余空格 去除尾部多余"."
                fields[i] = fields[i].strip().rstrip('.')
            # 将结果加入列表
            lines.append(fields)
    return lines


def choose_percent(lines: [], percent: int) -> []:
    """
    选择训练集百分比
    :param lines:
    :param percent:
    :return:
    """
    if percent >= 100:
        return lines

    if percent <= 0:
        return []

    n = int(len(lines) * (100-percent)/100)
    for i in range(0, n):
        x = random.randint(0, len(lines))
        del(lines[x])
    return lines


def choose_feature(lines: [], features: []) -> []:
    result = []
    for line in lines:
        new_line = []
        for i in range(0, len(features)):
            new_line.append(line[features[i]])
        result.append(new_line)
    return result


def div_param(lines: [], feature: int, n: int) -> []:
    for line in lines:
        line[feature] = int(int(line[feature]) / n)
    return lines


def get_field_category(line: []) -> int:
    """
    分类在数组的倒数第一个元素
    :param line:  line
    :return: int
    """
    return len(line) - 1


def count_prior(lines: []) -> {}:
    """
    计数分类数量（先验）
    :param lines: []
    :return: result: []

    result 格式:
    {'<=50K': 24720, '>50K': 7841}
    """
    result = {}

    # 获取分类在数组的索引
    field_category = get_field_category(lines[0])

    for line in lines:
        result.setdefault(line[field_category], 0)
        result[line[field_category]] += 1

    return result


def count_conditional(lines: []) -> {}:
    """
    计数条件数量（后验）
    :param lines: []
    :return:  result: {}

    result 格式:
    {'<=50K': [ {'39': 538, '50': 341, ......}, {'State-gov': 945, 'Self-emp-not-inc': 1817, ......}, ......] }
    {'>50K': [ {'39': 538, '50': 341, ......}, {'State-gov': 945, 'Self-emp-not-inc': 1817, ......}, ......] }
    """
    result = {}

    # 获取分类在数组的索引
    field_category = get_field_category(lines[0])

    for line in lines:
        result.setdefault(line[field_category], {})
        for column in range(len(line) - 1):
            result[line[field_category]].setdefault(column, {})
            result[line[field_category]][column].setdefault(line[column], 0)
            result[line[field_category]][column][line[column]] += 1

    return result


def calc_prior(prior: {}) -> {}:
    """
    计算先验概率
    :param prior: {} 
    :return:  result: {}

    result 格式:
    {'<=50K': 0.7591904425539756, '>50K': 0.2408095574460244}
    """
    result = {}

    total = 0
    for category in prior:
        total += prior[category]

    for category in prior:
        result.setdefault(category, 0.0)
        result[category] = prior[category] / total

    return result


def calc_conditional(conditional: {}, prior: {}) -> {}:
    """
    计算后验概率
    :param conditional: {}
    :param prior: {}
    :return: result: {}

    result 格式:
    {'<=50K': [
        {'39': 0.021763754045307445, '0.021763754045307445,': 0.013794498381877022, ......},
        {'State-gov': 0.03822815533980582, 'Self-emp-not-inc': 0.07350323624595469, ......},
        ......] }
    {'>50K': [
        {'39': 0.021763754045307445, '50': 0.013794498381877022, ......},
        {'State-gov': 945, 'Self-emp-not-inc': 1817, ......},
        ......] }
    """
    result = {}

    for category in conditional:
        result.setdefault(category, {})
        for columns in conditional[category]:
            result[category].setdefault(columns, {})
            for k, v in conditional[category][columns].items():
                result[category][columns][k] = v / prior[category]

    return result


def calc_test(test: [], prior: {}, conditional: {}, prior_count: {}) -> []:
    """
    计算测试值的概率
    :param test: []
    :param prior: {}
    :param conditional: {}
    :return: result: {}
    """
    result = []
    field_category = get_field_category(test[0])
    for line in test:
        row = {'result': line[field_category]}
        max = 0
        for category in conditional:
            prob = prior[category]
            for i in range(len(line) - 2):
                if line[i] in conditional[category][i]:
                    prob *= conditional[category][i][line[i]]
                else:  # 找不到分类时，使用拉普拉斯
                    laplace = 1 / (prior_count[category] + len(conditional[category][i]))
                    prob *= laplace
            row[category] = prob
            if prob > max:
                max = prob
                row['guess'] = category
        result.append(row)
    return result


def evaluate(result: []):
    """ 评估数据（打印精准率和召回率）
    :param result: 评估数组
    :return: 无需返回
    """
    positives = 0
    negatives = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for line in result:
        if line['result'] == '<=50K':
            positives += 1
            if 'guess' in line:
                if line['guess'] == line['result']:
                    true_positives += 1
                else:
                    false_positives += 1
        else:
            negatives += 1
            if 'guess' in line:
                if line['guess'] == line['result']:
                    true_negatives += 1
                else:
                    false_negatives += 1

    print('# 测试集: ' + str(positives + negatives))
    print('# 精准率: ' + str(true_positives / (true_positives + false_positives)))
    print('# 召回率: ' + str(true_positives / (true_positives + false_negatives)))
    print('-' * 40)


#######################################################################################################################
# 获取开始运行时间
start = time.process_time()

# 获取训练集：adult.txt
raw = read_file('adult.txt', ',')

# 获取测试集： adult.text
test = read_file('adult.test', ',')
#######################################################################################################################
# 训练集对分类的效果有什么影响
'''
# 选取 100% 的训练集数据训练分类模型
# 无需任何处理
# 训练集: 32561
# 测试条目 数量16281
# 精准率:0.7968636911942099
# 召回率:0.949683726279471
'''

'''
# 选取 50% 的训练集数据训练分类模型
# 用时反注释；用完后，记得注释
# raw = choose_percent(raw, 50)
# 训练集: 16281
# 测试条目 数量16281
# 精准率:0.7815842380377965
# 召回率:0.9520031344891762
'''

'''
# 选取 5% 的训练集数据训练分类模型
# 用时反注释；用完后，记得注释  
# raw = choose_percent(raw, 5)
# 训练集: 1629 (5%)
# 测试条目 数量16281
# 精准率:0.7524728588661037
# 召回率:0.9501421608448416
'''

'''
实验结论
随着训练集的减小，精准率降低，召回率变化不大
精准率在数据集5%的情况下，仍然有75%左右，训练集对于精准率波动的影响较小
'''
#######################################################################################################################
# 特征集对分类的效果有什么影响
# 以下数据：
# 训练集: 32561
# 测试集: 16281

# 分别选10个特征组合训练分类模型, 最后一个必须是CATEGORY
# 1
# features = [AGE, WORKCLASS, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCUPATION, RELATIONSHIP, RACE, SEX, CATEGORY]
# 精准率: 0.840691596300764
# 召回率: 0.9114210985178727
# 精准率提升6%，召回率下降3%

# 2
# features = [AGE, WORKCLASS, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCUPATION, RELATIONSHIP, RACE, COUNTRY, CATEGORY]
# 精准率: 0.840691596300764
# 召回率: 0.9114210985178727
# SEX -> COUNTRY，无变化，SEX可能没什么用

# 3
# features = [AGE, WORKCLASS, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCUPATION, RACE, SEX, HOURS, CATEGORY]
# 精准率: 0.869481302774427
# 召回率: 0.888341138772492
# 保留 SEX， COUNTRY -> HOURS， 精准率上升，召回率下降， COUNTRY有用，HOUR有用，SEX确定没什么用，不要性别歧视

# 4
# features = [AGE, WORKCLASS, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCUPATION, HOURS, RACE, COUNTRY, CATEGORY]
# 精准率: 0.8644149577804584
# 召回率: 0.8895233366434955
# SEX -> RACE， 变化不大， RACE可能也没什么用

# 5
# features = [AGE, WORKCLASS, EDUCATION, EDUCATION_NUM, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, COUNTRY, CATEGORY]
# 精准率: 0.8820265379975875
# 召回率: 0.9050998514606371
# RACE -> CAPITAL_GAIN， 精准率目前最高，召回率并非最高，但相对RACE的场合高，RACE确定没什么用, 不要种族歧视

# 6
# features = [AGE, WORKCLASS, EDUCATION, CAPITAL_LOSS, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, COUNTRY, CATEGORY]
# 精准率: 0.8989143546441496
# 召回率: 0.9112995271482146
# EDUCATION_NUM -> CAPITAL_LOSS，精准率和召回率都上升，EDUCATION_NUM可能没什么用

# 7
# features = [AGE, WORKCLASS, FNLWGT, CAPITAL_LOSS, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, COUNTRY, CATEGORY]
# 精准率: 0.7861680739847206
# 召回率: 0.941811175337187
# EDUCATION -> FNLWGT， 精准率狂降，召回率提升显著，EDUCATION有用

# 8
# features = [AGE, FNLWGT, EDUCATION, CAPITAL_LOSS, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, COUNTRY, CATEGORY]
# 精准率: 0.8139927623642943
# 召回率: 0.9406189015890717
# WORKCLASS -> FNLWGT, 精准率提升，召回略下降，WORKCLASS的作用比FNLWGT大

# 9
# features = [AGE, WORKCLASS, EDUCATION, CAPITAL_LOSS, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, HOURS, CATEGORY]
# 精准率: 0.8989143546441496
# 召回率: 0.9112995271482146
# COUNTRY -> HOURS, 精准率上升8，召回率下降3，HOURS的作用应该比COUNTRY大

# 10
# features = [AGE, WORKCLASS, EDUCATION, CAPITAL_LOSS, MARITAL_STATUS, COUNTRY, HOURS, CAPITAL_GAIN, HOURS, CATEGORY]
# 精准率: 0.9071974266184157
# 召回率: 0.90248
# OCCUPATION -> COUNTRY，精准率和召回率变化不大，两者效果应该类似

# 上面任选一条 features 反注释，并解开下面区域的注释块
''' 10大特征集测试
print('特征集:', str(len(features)))
raw = choose_feature(raw, features)
test = choose_feature(test, features)
'''

# 哪个特征集的分类效果最好
'''
AGE, WORKCLASS, EDUCATION, CAPITAL_LOSS, MARITAL_STATUS, OCCUPATION, HOURS, CAPITAL_GAIN, HOURS, CATEGORY
9号 特征集目前看效果最好（10号也差不多）
'''
#######################################################################################################################
# 连续特征以及未知特征的处理: age
# 注意，本实验不能使用变更后的特征集，只能使用原始数据
# 训练集: 32561
# 测试集: 16281

# 不分割
# 精准率: 0.7968636911942099
# 召回率: 0.949683726279471

# age 3
# raw = div_param(raw, AGE, 3)
# test = div_param(test, AGE, 3)
# 精准率: 0.7965420184961801
# 召回率: 0.9489365778884844

# age 5
# raw = div_param(raw, AGE, 5)
# test = div_param(test, AGE, 5)
# 精准率: 0.7971853638922396
# 召回率: 0.9493392070484582


# age 10
# raw = div_param(raw, AGE, 10)
# test = div_param(test, AGE, 10)
# 精准率: 0.7976678729392843
# 召回率: 0.9488234168739239

'''
精准率、召回率变化不大
'''
#######################################################################################################################
# 连续特征以及未知特征的处理: hour
# 注意，本实验不能使用变更后的特征集，只能使用原始数据
# 训练集: 32561
# 测试集: 16281

# 不分割
# 精准率: 0.7968636911942099
# 召回率: 0.949683726279471

# hour 3
# raw = div_param(raw, HOURS, 3)
# test = div_param(test, HOURS, 3)
# 精准率: 0.7961399276236429
# 召回率: 0.9494581375275727

# hour 5
# raw = div_param(raw, HOURS, 5)
# test = div_param(test, HOURS, 5)
# 精准率: 0.7962203457981504
# 召回率: 0.9503743520829334

# hour 10
# raw = div_param(raw, HOURS, 5)
# test = div_param(test, HOURS, 5)
# 精准率: 0.7962203457981504
# 召回率: 0.9503743520829334

#######################################################################################################################
# 投资收益与投资损失
# 注意，本实验不能使用变更后的特征集，只能使用原始数据
# 训练集: 32561
# 测试集: 16281

# 不分割
# 精准率: 0.7968636911942099
# 召回率: 0.949683726279471

# 1000
# raw = div_param(raw, CAPITAL_GAIN, 1000)
# test = div_param(test, CAPITAL_GAIN, 1000)
# 精准率: 0.7918777643747487
# 召回率: 0.9470090402000385


# 无、低、高
def get_mean(lines: [], feature: int) -> int:
    """
    获取平均值
    :param lines:
    :param feature:
    :return:
    """
    total = 0
    for line in lines:
        total += int(line[feature])
    mean = total / len(lines)
    return int(mean)


def set_capital(lines: [], gain_mean: int, loss_mean: int) -> []:
    """
    根据平均值，设置无、低、高
    :param lines:
    :param gain_mean:
    :param loss_mean:
    :return:
    """
    for line in lines:
        if int(line[CAPITAL_GAIN]) >= gain_mean:
            line[CAPITAL_GAIN] = 'high'
        elif line[CAPITAL_GAIN] == '0':
            line[CAPITAL_GAIN] = 'none'
        else:
            line[CAPITAL_GAIN] = 'low'

        if int(line[CAPITAL_LOSS]) >= loss_mean:
            line[CAPITAL_LOSS] = 'high'
        elif line[CAPITAL_LOSS] == '0':
            line[CAPITAL_LOSS] = 'none'
        else:
            line[CAPITAL_LOSS] = 'low'
    return lines


'''
gain_mean = get_mean(raw, CAPITAL_GAIN)
loss_mean = get_mean(raw, CAPITAL_LOSS)
raw = set_capital(raw, gain_mean, loss_mean)
test = set_capital(test, gain_mean, loss_mean)
'''
# 精准率: 0.7799758745476477
# 召回率: 0.9414676761793827

#######################################################################################################################
# 未知数据的处理
# 训练集: 30162
# 测试集: 15060


# 忽略'?'
def clear_unknown(lines: []) -> []:
    result = []
    for line in lines:
        found = False
        for i in range(len(line)):
            if line[i] == '?':
                found = True
                break
        if not found:
            result.append(line)
    return result


'''
raw = clear_unknown(raw)
test = clear_unknown(test)
'''
# 精准率: 0.7911971830985915
# 召回率: 0.9470024233484353


# 补全
def found_unknow(lines: []) -> {}:
    """
    检查哪些字段有未知数据，各几条
    :param lines:
    :return:
    """
    unknown = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9:0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
    for line in lines:
        for i in range(len(line)):
            if line[i] == '?':
                unknown[i] += 1
    return unknown

'''
raw_unknown = found_unknow(raw)
test_unknown = found_unknow(test)
print(raw_unknown)
print(test_unknown)
'''
# 即：
# WORKCLASS、OCCUPATION、COUNTRY 有未知数据
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
#   Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
#   Protective-serv, Armed-Forces
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India,
#   Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal,
#   Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua,
#   Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


# 用相应的方式补全
# 用众数补全
def set_unknown(lines: [], is_raw: bool) -> []:
    for line in lines:
        for i in range(len(line)):
            if line[WORKCLASS] == '?':
                line[WORKCLASS] = 'Private'
            if line[COUNTRY] == '?':
                line[COUNTRY] = 'United-States'
            if line[OCCUPATION] == '?':
                if is_raw:
                    line[OCCUPATION] = 'Adm-clerical'
                else:
                    line[OCCUPATION] = 'xec-managerial'
    return lines


'''
raw = set_unknown(raw, True)
test = set_unknown(test, False)
'''
# 精准率: 0.7900281463610777
# 召回率: 0.9504643962848297
#######################################################################################################################
# 计数分类数量
prior_count = count_prior(raw)

# 计数分类数量
conditional_count = count_conditional(raw)
# print(conditional_count['<=50K'][WORKCLASS])  # 众数是 Private
# print(conditional_count['>50K'][WORKCLASS])  # 众数是 Private
# print(conditional_count['<=50K'][OCCUPATION])  # 众数是 Adm-clerical
# print(conditional_count['>50K'][OCCUPATION])  # # 众数是 Exec-managerial
# print(conditional_count['<=50K'][COUNTRY])  # 众数是 United-States
# print(conditional_count['>50K'][COUNTRY])  # 众数是 United-States
# 计算先验概率
prior_prob = calc_prior(prior_count)

# 计算后验概率
conditional_prob = calc_conditional(conditional_count, prior_count)

# 根据测试集进行预测
test_calc = calc_test(test, prior_prob, conditional_prob, prior_count)

# 根据预测结果进行分析精准率和召回率
print('# 训练集:', str(len(raw)))
evaluate(test_calc)

# 获取运行结束时间
end = time.process_time()
print("运行时间：", end - start)
