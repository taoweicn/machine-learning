import numpy as np


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def stochastic_gradient_ascent(matrix, class_labels, n_iters=150):
    """
    随机梯度上升算法
    """
    m, n = np.shape(matrix)
    weights = np.ones(n)
    for i in range(n_iters):
        index = range(m)
        for j in range(m):
            alpha = 4 / (1.0 + i + j) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            rand_index = int(np.random.uniform(0, len(index)))
            h = sigmoid(sum(matrix[index[rand_index]] * weights))
            error = class_labels[index[rand_index]] - h
            weights = weights + alpha * error * matrix[index[rand_index]]
            # del (index[rand_index])
    return weights


def classify(X, weights):
    prob = sigmoid(sum(X * weights))
    return 1 if prob > 0.5 else 0


def main():
    with open('data/horseColicTraining.txt', 'r') as f:
        train = np.array([[float(word) for word in line.strip().split('\t')] for line in f.readlines()])
    X_train = train[:, :-1]
    y_train = train[:, -1]

    with open('data/horseColicTest.txt', 'r') as f:
        test = np.array([[float(word) for word in line.strip().split('\t')] for line in f.readlines()])
    X_test = test[:, :-1]
    y_test = test[:, -1]

    # sklearn
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)
    print('使用sklearn LogisticRegression准确率：', log_reg.score(X_test, y_test))

    # 随机梯度上升法
    train_weights = stochastic_gradient_ascent(X_train, y_train)
    y_predict = np.array([classify(vect, train_weights) for vect in X_test])
    error_rate = sum(y_test == y_predict) / len(y_test)
    print('使用随机梯度上升法准确率：', error_rate)


if __name__ == '__main__':
    main()
