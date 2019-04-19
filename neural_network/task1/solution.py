import numpy as np
from sklearn.linear_model import LogisticRegression


def main():
    with open('data/horseColicTraining.txt', 'r') as f:
        train = np.array([[float(word) for word in line.strip().split('\t')] for line in f.readlines()])
    X_train = train[:, :-1]
    y_train = train[:, -1]

    with open('data/horseColicTest.txt', 'r') as f:
        test = np.array([[float(word) for word in line.strip().split('\t')] for line in f.readlines()])
    X_test = test[:, :-1]
    y_test = test[:, -1]

    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)
    print(log_reg.score(X_test, y_test))


if __name__ == '__main__':
    main()
