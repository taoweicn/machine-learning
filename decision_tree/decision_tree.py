import math
import numpy as np
from collections import Counter
from typing import List


class DecisionTree:
    __X_train: np.array = None
    __y_train: np.array = None
    __decision_tree: dict = None

    def fit(self, data: list):
        dataset = np.array(data)
        self.__X_train = dataset[:, :-1]
        self.__y_train = dataset[:, -1]
        return self

    @staticmethod
    def __calc_entropy(X_train: np.array, y_train: np.array) -> float:
        label_counts = Counter(y_train)
        # 计算信息熵
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / len(X_train)
            entropy -= prob * math.log(prob, 2)
        return entropy

    @staticmethod
    def __split_dataset(X_train: np.array, y_train: np.array, index: int, value: any) -> (np.array, np.array):
        new_index = [row for row, data in enumerate(X_train) for i, v in enumerate(data) if i == index and v == value]
        new_X_train = np.array([data for i, data in enumerate(X_train) if i in new_index])
        new_y_train = np.array([data for i, data in enumerate(y_train) if i in new_index])
        return new_X_train, new_y_train

    def __choose_best_feature(self, X_train: np.array, y_train: np.array) -> int:
        features_num = X_train.shape[1]
        base_entropy = self.__calc_entropy(X_train, y_train)
        # 最优的信息增益值, 和最优的特征的编号
        best_gain, best_feature = 0.0, -1
        # 计算按照各个特征分类的信息熵
        for i in range(features_num):
            feature_list = set(X_train[:, i])
            new_entropy = 0.0
            for feature in feature_list:
                sub_X_train, sub_y_train = self.__split_dataset(X_train, y_train, i, feature)
                prob = len(sub_X_train) / float(len(X_train))
                new_entropy += prob * self.__calc_entropy(sub_X_train, sub_y_train)
            info_gain = base_entropy - new_entropy
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = i
        return best_feature

    def __predict(self, tree: dict, test_data: list, labels: List[str]) -> any:
        root = list(tree.keys())[0]
        value = tree[root]
        root_index = labels.index(root)
        key = test_data[root_index]
        feat_value = value[key]
        # 判断分支是否结束
        if isinstance(feat_value, dict):
            class_label = self.__predict(feat_value, test_data, labels)
        else:
            class_label = feat_value
        return class_label

    def predict(self, test_data: list, labels: list) -> List[str]:
        if not self.__decision_tree:
            self.build_tree(labels)
        return [self.__predict(self.__decision_tree, data, labels) for data in test_data]

    @staticmethod
    def score(y_predict: list, y_true: list) -> float:
        return sum(np.array(y_predict) == np.array(y_true)) / len(y_true)

    def __build_tree(self, X_train: np.array, y_train: np.array, labels: List[str]) -> dict or str:
        if Counter(y_train)[y_train[0]] == len(y_train):
            return y_train[0]
        if len(X_train[0]) == 1:
            major_label = Counter(y_train).most_common(1)[0]
            return major_label

        best_feat = self.__choose_best_feature(X_train, y_train)
        best_feat_label = labels[best_feat]
        tree = {best_feat_label: {}}
        feature_list = set(X_train[:, best_feat])
        for value in feature_list:
            # 遍历当前选择特征包含的所有属性
            tree[best_feat_label][value] = self.__build_tree(
                *self.__split_dataset(X_train, y_train, best_feat, value),
                labels
            )
        return tree

    def build_tree(self, labels: List[str]) -> dict:
        self.__decision_tree = self.__build_tree(self.__X_train, self.__y_train, labels)
        return self.__decision_tree

    def __draw_tree(self, tree: dict, depth: int) -> None:
        if not tree:
            return
        if isinstance(tree, str):
            print(depth, '-' * 2 * depth, tree)
            return
        for key in tree:
            print(depth, '-' * 2 * depth, key)
            self.__draw_tree(tree[key], depth + 1)

    def draw_tree(self, tree):
        return self.__draw_tree(tree, 0)
