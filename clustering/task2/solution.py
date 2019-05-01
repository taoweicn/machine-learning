import numpy as np
from itertools import groupby


def load_data(filename):
    with open(filename, 'r', encoding='iso-8859-1') as f:
        data = [line.strip().split(',') for line in f.readlines()]
    return data


def split_data(dataset):
    # 求所有训练数据中BSSID的并集
    bssids = list(set(np.array(dataset)[:, 0]))
    feature_num = len(bssids)

    def generate_feature(data):
        labels = data[0]
        dataset = data[1:]
        X = []
        y = []
        # 按指纹分组
        train_data_group = groupby(dataset, lambda x: x[-1])
        for fin, group in train_data_group:
            feature = [-100] * feature_num
            room_label = 0
            group = list(group)
            for line in group:
                # 再做一次判断是因为测试数据集中可能出现训练数据集中没有的BSSID
                if line[0] in bssids:
                    feature[bssids.index(line[0])] = int(line[1])
                    room_label = int(line[2])
            X.append(feature)
            y.append(room_label)
        return X, y, labels

    return generate_feature(dataset)


def reduce_dimension(X, n):
    # 使用MDS进行降维
    from sklearn.manifold import MDS
    embedding = MDS(n_components=n)
    return embedding.fit_transform(X)


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score

    X, y, labels = split_data(load_data('data/DataSetKMeans1.csv'))
    X = reduce_dimension(X, 5)
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n).fit(X)
        print('DBI:', davies_bouldin_score(X, kmeans.labels_))

    X, y, labels = split_data(load_data('data/DataSetKMeans2.csv'))
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n).fit(X)
        print('DBI:', davies_bouldin_score(X, kmeans.labels_))


if __name__ == '__main__':
    main()
