import csv
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename) as f:
        data = csv.reader(f)
        header = next(data)
        has_y = len(header) > 30
        X = []
        y = []
        for line in data:
            X.append([float(line[i]) for i in range(30)])
            if has_y:
                y.append([float(line[30]), float(line[31])])
        return (X, y) if has_y else X


def merge(X):
    """
    合并两种频率的信号强度，采用求平均值的方法
    """
    length = len(X)
    matrix = [None] * length
    for index, line in enumerate(X):
        new_len = len(line) // 2
        new_line = [None] * new_len
        for i in range(new_len):
            new_line[i] = (line[i] + line[i + new_len]) / 2
        matrix[index] = new_line
    return matrix


def split(X):
    """
    分开两种频率的信号强度，分开为两个样本
    """
    length = len(X)
    matrix = [None] * (length * 2)
    for index, line in enumerate(X):
        half_len = len(line) // 2
        matrix[index] = line[0:half_len]
        matrix[index + length] = line[half_len:]
    return matrix


def calc_aver_error(list1, list2) -> float:
    """
    计算平均定位误差
    """
    assert len(list1) == len(list2)
    num = len(list1)
    error_sum = 0
    for i in range(num):
        error_sum += np.linalg.norm(np.array(list1[i]) - np.array(list2[i]))
    return error_sum / num


def is_covered(rss: float) -> bool:
    """
    是否被覆盖的标准
    :param rss: 信号强度
    :return: 是否被覆盖
    """
    return rss > -105


def calc_coverage_num(matrix):
    """
    计算每个坐标点被多少个sector覆盖
    """
    nums = []
    for line in matrix:
        num = 0
        for i in range(len(line) // 2):
            if is_covered(line[i]) or is_covered(line[i + len(line) // 2]):
                num += 1
        nums.append(num)
    return nums


def calc_coverage_rate(arr):
    """
    计算样本覆盖率
    """
    return reduce(lambda acc, num: acc + 1 if num > 0 else acc, calc_coverage_num(arr), 0) / len(arr)


def calc_sector_coverage_rate(matrix):
    """
    计算所有sector的覆盖率
    """
    sector_num = len(matrix[0]) // 2
    rate = [0] * sector_num
    for i in range(sector_num):
        for j in range(len(matrix)):
            if is_covered(matrix[j][i]) or is_covered(matrix[j][i + sector_num]):
                rate[i] += 1
        rate[i] /= len(matrix)
    return rate


def calc_score(X, y, estimator):
    """
    采用train_test_split计算得分，越小越好
    """
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    estimator.fit(X_train, y_train)
    y_predict = estimator.predict(X_test)
    return calc_aver_error(y_test, y_predict)


def get_min_sector(matrix):
    """
    计算能保持覆盖率 > 95%的最少sector个数
    """
    sector_num = len(matrix[0]) // 2
    while sector_num > 0:
        sector_rate = calc_sector_coverage_rate(matrix)
        sorted_sector_rate = np.argsort(sector_rate)
        index = sorted_sector_rate[0]
        print(index)
        matrix = np.delete(matrix, [index, index + sector_num], axis=1)
        print(matrix[0])
        sector_num -= 1
        print(calc_coverage_rate(matrix))
        if calc_coverage_rate(matrix) <= 0.95:
            return sector_num + 1


def get_min_matrix(matrix):
    """
    返回删除一些sector后的数据集
    """
    new_num = get_min_sector(matrix)
    sector_rate = calc_sector_coverage_rate(matrix)
    sorted_sector_rate = np.argsort(sector_rate)
    sector_num = len(sector_rate)
    print(sorted_sector_rate)
    for index in sorted_sector_rate:
        if new_num == sector_num:
            break
        matrix = np.delete(matrix, [index, index + sector_num], axis=1)
        sector_num -= 1
    return matrix


def dfs(raw_matrix, matrix, last, indexes, res):
    if calc_coverage_rate(matrix) <= 0.95:
        indexes[last] = 1
        return
    if sum(res) > sum(indexes):
        res = indexes[:]
        print(sum(res))
    length = len(indexes)
    for i in range(length):
        if indexes[i] == 1:
            indexes[i] = 0
            delete_index = np.array(
                reduce(lambda acc, item: acc + [item[0]] if item[1] == 0 else acc, enumerate(indexes), [])
            )
            new_matrix = np.delete(raw_matrix, np.append(delete_index, delete_index + length), axis=1)
            dfs(raw_matrix, new_matrix, i, indexes, res)
            indexes[i] = 1


def delete_sector(matrix, delete_indexes):
    delete_indexes = np.array(delete_indexes)
    num = len(matrix[0]) // 2
    delete_indexes = np.append(delete_indexes, delete_indexes + num)
    matrix = np.delete(matrix, delete_indexes, axis=1)
    print(calc_coverage_rate(matrix))
    return matrix


def binarize(X, y, estimator):
    from sklearn import preprocessing
    scores = []
    thresholds = []
    scores.append(calc_score(X, y, estimator))
    thresholds.append('None')
    for threshold in [-105, -126.23]:
        binarizer = preprocessing.Binarizer(threshold=threshold)
        scores.append(calc_score(binarizer.transform(X), y, estimator))
        thresholds.append(str(threshold))
    plt.scatter(thresholds, scores)
    plt.xlabel('thresholds')
    plt.ylabel('scores')
    plt.show()


def data_processing(X, y, estimator):
    scores = [
        calc_score(X, y, estimator),
        calc_score(merge(X), y, estimator),
        calc_score(split(X), y * 2, estimator)
    ]
    methods = ['raw', 'merge', 'split']
    plt.scatter(methods, scores)
    plt.xlabel('methods')
    plt.ylabel('scores')
    plt.show()


def main():
    X_train, y_train = load_data('data/dataAll.csv')
    # res = [1] * 15
    # dfs(X_train, X_train, None, [1] * 15, res)
    # print(res)
    # return
    # delete_sector(X_train, [1, 3, 6, 8, 10, 11, 13, 14])
    # return

    scores = []
    models = []

    # 随机森林
    from sklearn.ensemble import RandomForestRegressor
    # from sklearn.model_selection import GridSearchCV
    # param_grid = [
    #     {
    #         'bootstrap': [True, False],
    #         # 'oob_score': [True, False],
    #         'n_estimators': [i for i in range(10, 110, 10)]
    #     }
    # ]
    # grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    models.append('RandomForest')
    scores.append(calc_score(X_train, y_train, RandomForestRegressor(n_estimators=100)))

    # kNN
    from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.model_selection import GridSearchCV
    # param_grid = [
    #     {
    #         'n_neighbors': [i for i in range(1, 30)]
    #     }
    # ]
    # grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    models.append('kNN')
    scores.append(calc_score(X_train, y_train, KNeighborsRegressor(n_jobs=-1, n_neighbors=2)))

    # 梯度提升
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    models.append('GradientBoosting')
    scores.append(
        calc_score(X_train, y_train, MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=10)))
    )

    # 神经网络多层感知器
    from sklearn.neural_network import MLPRegressor
    models.append('NeuralNetwork')
    scores.append(calc_score(X_train, y_train, MLPRegressor(hidden_layer_sizes=(100, 100))))

    # SVM
    from sklearn import svm
    clf_x = svm.SVR(C=1000, gamma=0.01)
    clf_y = svm.SVR(C=1000, gamma=0.01)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
    y_train = np.array(y_train)
    clf_x.fit(X_train, y_train[:, 0])
    clf_y.fit(X_train, y_train[:, 1])
    x = clf_x.predict(X_test)
    y = clf_y.predict(X_test)
    models.append('SVM')
    scores.append(calc_aver_error(y_test, np.column_stack((x, y))))

    # 线性回归
    from sklearn.linear_model import LinearRegression
    models.append('LinearRegression')
    scores.append(calc_score(X_train, y_train, LinearRegression()))

    # 贝叶斯岭回归
    from sklearn.linear_model import BayesianRidge
    from sklearn.multioutput import MultiOutputRegressor
    models.append('BayesianRidge')
    scores.append(calc_score(X_train, y_train, MultiOutputRegressor(BayesianRidge())))

    plt.scatter(models, scores)
    plt.xlabel('models')
    plt.ylabel('scores')
    plt.show()

    # y_predict = clf.predict(X_test)
    # with open('data/predict.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for index, coordinate in enumerate(y_predict):
    #         writer.writerow([index + 1] + list(coordinate))


if __name__ == '__main__':
    main()
