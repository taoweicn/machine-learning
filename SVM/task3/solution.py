from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def load_data(filename):
    data_dict = loadmat(filename)
    return data_dict['X'], data_dict['y']


def main():
    X_train, y_train = load_data('data/task3_train.mat')
    y_train = list(map(lambda x: x[0], y_train))
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print(svc.score(X_test, y_test))


if __name__ == '__main__':
    main()
