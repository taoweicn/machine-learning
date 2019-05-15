import csv
import numpy as np
from itertools import groupby


# import warnings
# warnings.filterwarnings('ignore')


def load_data(filename):
    with open(filename, 'r', encoding='iso-8859-1') as f:
        return list(csv.reader(f))


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

    clusters_num = range(2, 10)

    X, y, labels = split_data(load_data('data/DataSetKMeans1.csv'))
    X = reduce_dimension(X, 5)
    print("dataset1")
    print("clusters_num:", ', '.join(map(str, clusters_num)))
    print_str = "DBI:"
    for n in clusters_num:
        kmeans = KMeans(n_clusters=n).fit(X)
        print_str += str(' %.2f' % davies_bouldin_score(X, kmeans.labels_))
    print(print_str, '\n')

    X, y, labels = split_data(load_data('data/DataSetKMeans2.csv'))
    X = reduce_dimension(X, 5)
    print("dataset2")
    print("clusters_num:", ', '.join(map(str, clusters_num)))
    print_str = "DBI:"
    for n in clusters_num:
        kmeans = KMeans(n_clusters=n).fit(X)
        print_str += str(' %.2f' % davies_bouldin_score(X, kmeans.labels_))
    print(print_str, '\n')


if __name__ == '__main__':
    main()
