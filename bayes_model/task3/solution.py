import numpy as np


def one_hot_encoder(feature_num, data):
    feature = np.zeros(feature_num, dtype=np.int)
    for num in data:
        feature[int(num)] = 1
    return feature


def main():
    with open('data/train/train_data.txt', 'r') as f:
        X_train = [one_hot_encoder(10000, line.strip().split(' ')) for line in f.readlines()]

    with open('data/train/train_labels.txt', 'r') as f:
        y_train = [int(line.strip()) for line in f.readlines()]

    from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
    from sklearn.ensemble import VotingClassifier
    # from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import train_test_split

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

    # param_grid = [{
    #     'fit_prior': (True, False),
    #     'alpha': [i * 0.1 for i in range(9, 12)]
    # }]
    # clf = MultinomialNB()
    # grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_estimator_)

    voting_clf = VotingClassifier(estimators=[
        ('MultinomialNB', MultinomialNB()),
        ('GaussianNB', GaussianNB()),
        ('BernoulliNB', BernoulliNB()),
        ('ComplementNB', ComplementNB())],
        voting='hard',
        n_jobs=-1
    )
    voting_clf.fit(X_train, y_train)
    # print(voting_clf.score(X_test, y_test))

    with open('data/test/test_data.txt', 'r') as f:
        X_test = [one_hot_encoder(10000, line.strip().split(' ')) for line in f.readlines()]

    y_predict = voting_clf.predict(X_test)
    with open('data/test/predict_data.txt', 'w') as f:
        f.write('\n'.join([str(num) for num in y_predict]))


if __name__ == '__main__':
    main()
