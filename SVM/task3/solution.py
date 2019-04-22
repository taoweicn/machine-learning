from scipy.io import loadmat
from sklearn.svm import LinearSVC


def load_data(filename):
    data_dict = loadmat(filename)
    return data_dict['X'], data_dict['y']


def calc_score(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    svc = LinearSVC(dual=False)
    svc.fit(X_train, y_train)
    print('score:', svc.score(X_test, y_test))


def main():
    X_train, y_train = load_data('data/task3_train.mat')
    y_train = list(map(lambda x: x[0], y_train))
    calc_score(X_train, y_train)
    svc = LinearSVC(dual=False)
    svc.fit(X_train, y_train)
    X_test = loadmat('data/task3_test.mat')['X']
    y_predict = svc.predict(X_test)
    with open('data/predict.txt', 'w') as f:
        f.write('\n'.join([str(num) for num in y_predict]))


if __name__ == '__main__':
    main()
