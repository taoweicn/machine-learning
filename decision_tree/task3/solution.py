import numpy as np
from sklearn.tree import DecisionTreeClassifier


def one_hot_encoder(feature_num, data):
    feature = np.zeros(feature_num, dtype=np.int)
    for num in data:
        feature[int(num)] = 1
    return feature


def main():
    with open('data/train/train_data.txt', 'r') as f:
        X_train = [one_hot_encoder(10000, line.strip().split(' ')) for line in f.readlines()]

    with open('data/train/train_labels.txt', 'r') as f:
        y_train = np.array([int(line.strip()) for line in f.readlines()])

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
    dt_clf = DecisionTreeClassifier(criterion='gini')
    dt_clf.fit(X_train, y_train)
    # print(dt_clf.score(X_test, y_test))

    with open('data/train/train_data.txt', 'r') as f:
        X_test = [one_hot_encoder(10000, line.strip().split(' ')) for line in f.readlines()]

    y_predict = dt_clf.predict(X_test)
    with open('data/test/predict_data.txt', 'w') as f:
        f.write('\n'.join([str(num) for num in y_predict]))


if __name__ == '__main__':
    main()
