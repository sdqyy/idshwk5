# -*- coding = utf-8 -*-
# @Time : 2022/5/5 16:32
# @Author : 秦云杨
# @file : test.py
# @Software : PyCharm

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist = []


class Domain:
    def __init__(self, _name, _domain_name, _numbers, _entropy, _seg, _label):
        self.name = _name
        self.domain_len = _domain_name
        self.numbers = _numbers
        self.entropy = _entropy
        self.seg = _seg
        self.label = _label

    def returnData(self):
        return [self.domain_len, self.numbers, self.entropy, self.seg]

    def returnLabel(self):
        if self.label == "dga":
            return 0
        else:
            return 1


def cal_num(str):
    num = 0
    for i in str:
        if i.isdigit():
            num += 1
    return num


def cal_seg(str):
    num = 0
    for i in str:
        if i == '.':
            num += 1
    return num


def calEntropy(string):
    h = 0.0
    sumt = 0
    letter = [0] * 26
    string = string.lower()
    for i in range(len(string)):
        if string[i].isalpha():
            letter[ord(string[i]) - ord('a')] += 1
            sumt += 1
    # print('\n', letter)
    for i in range(26):
        p = 1.0 * letter[i] / sumt
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


def initData(filename, domainlist):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            if len(tokens) > 1:
                label = tokens[1]
            else:
                label = "none"
            domain_len = len(name)
            numbers = cal_num(name)
            entropy = calEntropy(name)
            seg = cal_seg(name)
            domainlist.append(Domain(name, domain_len, numbers, entropy, seg, label))


def main():
    initData("train.txt", domainlist)
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)

    domainlist2 = []
    initData("test.txt", domainlist2)
    with open("result.txt", "w") as f:
        for i in domainlist2:
            f.write(i.name)
            f.write(",")
            if clf.predict([i.returnData()])[0] == 0:
                f.write("notdga")
            else:
                f.write("dga")
            f.write("\n")

    # print(clf.predict([[3600, 10000, 3]]))
    # print(clf.predict([[3600, 3600, 1]]))
    # print(clf.predict([[100, 100, 3]]))
    # print(clf.predict([[100, 100, 1]]))


if __name__ == '__main__':
    main()
