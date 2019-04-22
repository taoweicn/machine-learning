import csv
import numpy as np
from functools import reduce


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


def calc_aver_error(list1, list2):
    """
    计算平均定位误差
    """
    assert len(list1) == len(list2)
    num = len(list1)
    error_sum = 0
    for i in range(num):
        error_sum += np.linalg.norm(np.array(list1[i]) - np.array(list2[i]))
    return error_sum / num


def calc_coverage_num(arr):
    nums = []
    for li in arr:
        num = 0
        for i in range(15):
            if li[i] > -105 or li[i + 15] > -105:
                num += 1
        nums.append(num)
    return nums


def calc_coverage_rate(arr):
    return reduce(lambda acc, num: acc + 1 if num > 0 else acc, calc_coverage_num(arr), 0) / len(arr)


def calc_score(X, y, estimator):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    estimator.fit(X_train, y_train)
    y_predict = estimator.predict(X_test)
    return calc_aver_error(y_test, y_predict)


def main():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    X_train, y_train = load_data('data/dataAll.csv')
    param_grid = [
        {
            'bootstrap': [True, False],
            # 'oob_score': [True, False],
            'n_estimators': [i for i in range(10, 110, 10)]
        }
    ]
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    rfr = RandomForestRegressor(n_estimators=100)
    rfr.fit(X_train, y_train)
    X_test = load_data('data/testAll.csv')
    y_predict = rfr.predict(X_test)
    with open('data/predict.csv', 'w') as f:
        writer = csv.writer(f)
        for index, coordinate in enumerate(y_predict):
            writer.writerow([index + 1] + list(coordinate))


if __name__ == '__main__':
    main()
