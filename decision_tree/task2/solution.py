import numpy as np
from itertools import groupby


def main():
    with open('data/TrainDT.csv', 'r', encoding='iso-8859-1') as f:
        train_data = [line.strip().split(',') for line in f.readlines()]

    with open('data/TestDT.csv', 'r', encoding='iso-8859-1') as f:
        test_data = [line.strip().split(',') for line in f.readlines()]

    # 求所有训练数据中BSSID的并集
    bssids = list(set(np.array(train_data)[:, 0]))
    feature_num = len(bssids)

    def generate_feature(data):
        labels = data[0]
        dataset = data[1:]
        X = []
        y = []
        # 按指纹分组
        train_data_group = groupby(dataset, lambda x: x[-1])
        for fin, group in train_data_group:
            feature = [0] * feature_num
            room_label = 0
            group = list(group)
            for line in group:
                # 再做一次判断是因为测试数据集中可能出现训练数据集中没有的BSSID
                if line[0] in bssids:
                    feature[bssids.index(line[0])] = 1
                    room_label = int(line[2])
            X.append(feature)
            y.append(room_label)
        return X, y, labels

    X_train, y_train, labels = generate_feature(train_data)
    X_test, y_test, labels = generate_feature(test_data)
    from sklearn.tree import DecisionTreeClassifier
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    print('score:', dt_clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
