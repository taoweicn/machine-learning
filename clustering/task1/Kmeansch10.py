# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:57:43 2018

@author: Administrator
"""

from numpy import *
import matplotlib.pyplot as plt


'''
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        A=list(fltLine) #与2.x区别
        dataMat.append(A)
    return dataMat
'''


def dist_eclud(vec_a, vec_b):
    return sqrt(sum(power(vec_a - vec_b, 2)))


# 随机生成簇中心函数
def rand_cent(dataset, k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_j = min(dataset[:, j])
        range_j = float(max(dataset[:, j]) - min_j)
        centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids


# dataSet为数据集，k为分簇数目，dist_eclud为距离函数，rand_cent为随机选择簇中心方法
def k_means(dataset, k, dist_meas=dist_eclud, create_cent=rand_cent):
    m = shape(dataset)[0]
    # 初始化矩阵cluster_assment，第1列记录簇索引值，第2列存储误差
    cluster_assment = mat(zeros((m, 2)))
    # 初始化簇中心
    centroids = create_cent(dataset, k)
    # 标志变量，用于判断是否继续迭代
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # 将每个样本点分配到与其最近的簇中心所在的簇
        for i in range(m):
            min_dist = inf
            min_index = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :], dataset[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            # 如果样本被划分到不同的簇，则改变标志变量，表示需要继续迭代
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
        # 打印簇中心
        print(centroids)
        # 由于样本划分发生变化，因此需要重新计算簇中心
        for cent in range(k):
            # 提取处属于同一簇的所有样本
            pts_in_clust = dataset[nonzero(cluster_assment[:, 0].A == cent)[0]]
            # 计算不同簇所有样本的平均值作为簇中心
            centroids[cent, :] = mean(pts_in_clust, axis=0)
    return centroids, cluster_assment


# dataset为数据集，k为分簇数目，dist_eclud为距离函数
def bi_k_means(dataset, k, dist_meas=dist_eclud):
    m = shape(dataset)[0]
    cluster_assment = mat(zeros((m, 2)))
    # 将所有样本的均值作为簇中心
    centroid0 = mean(dataset, axis=0).tolist()[0]
    # 创建簇中心列表
    cent_list = [centroid0]
    # 计算每个样本的误差
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(mat(centroid0), dataset[j, :]) ** 2
    while len(cent_list) < k:
        lowest_sse = inf
        # 拆分每个簇，并计算拆分后的SSE，选择拆分后SSE最小的簇，保存拆分
        for i in range(len(cent_list)):
            pts_in_curr_cluster = dataset[nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_clust_ass = k_means(pts_in_curr_cluster, 2, dist_meas)
            sse_split = sum(split_clust_ass[:, 1])
            sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print("sse_split, and notSplit: ", sse_split, sse_not_split)
            if (sse_split + sse_not_split) < lowest_sse:
                best_cent_to_split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        # 一个簇拆分为二后，其中一个簇新增加簇索引，另一个保存原簇索引号
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print('the best_cent_to_split is: ', best_cent_to_split)
        print('the len of best_clust_ass is: ', len(best_clust_ass))
        # 重置簇中心
        cent_list[best_cent_to_split] = best_new_cents[0, :].tolist()[0]
        cent_list.append(best_new_cents[1, :].tolist()[0])
        # 调整样本的簇索引号及误差
        cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
    return mat(cent_list), cluster_assment


# 根据经纬度计算球面距离，vecA[0,：]表示A点经纬度
def dist_SLC(vec_a, vec_b):
    a = sin(vec_a[0, 1] * pi / 180) * sin(vec_b[0, 1] * pi / 180)
    b = cos(vec_a[0, 1] * pi / 180) * cos(vec_b[0, 1] * pi / 180) * \
        cos(pi * (vec_b[0, 0] - vec_a[0, 0]) / 180)
    return arccos(a + b) * 6371.0


# num_clust为簇数目
def cluster_clubs(num_clust=5):
    dat_list = []
    # 导入数据
    for line in open('data/places.txt').readlines():
        line_arr = line.split('\t')
        dat_list.append([float(line_arr[4]), float(line_arr[3])])
    dat_mat = mat(dat_list)
    # 采用二分k-均值算法进行聚类
    my_centroids, clust_assing = bi_k_means(dat_mat, num_clust, dist_meas=dist_SLC)
    # 定义画布，背景
    fig = plt.figure()
    rect = [0.0, 0.0, 1.0, 1.0]
    # 不同图形标识
    scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    # 导入地图
    img_p = plt.imread('Portland.png')
    ax0.imshow(img_p)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 采用不同图形标识不同簇
    for i in range(num_clust):
        pts_in_curr_cluster = dat_mat[nonzero(clust_assing[:, 0].A == i)[0], :]
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax1.scatter(
            pts_in_curr_cluster[:, 0].flatten().A[0],
            pts_in_curr_cluster[:, 1].flatten().A[0],
            marker=marker_style,
            s=90
        )
    # 采用‘+’表示簇中心
    ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


def main():
    cluster_clubs()


if __name__ == '__main__':
    main()
