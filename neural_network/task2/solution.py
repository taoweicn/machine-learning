import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


def main():
    with open('data/train/train_texts.dat', 'rb') as f:
        train_text = pickle.load(f)
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors_train = vectorizer.fit_transform(train_text)

    with open('data/train/train_labels.txt', 'r') as f:
        y_train = [int(line.strip()) for line in f.readlines()]

    clf = MLPClassifier()
    clf.fit(vectors_train, y_train)

    with open('data/test/test_texts.dat', 'rb') as f:
        test_text = pickle.load(f)
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors_test = vectorizer.fit_transform(test_text)

    y_predict = clf.predict(vectors_test)
    with open('data/test/predict_data.txt', 'w') as f:
        f.write('\n'.join([str(num) for num in y_predict]))


if __name__ == '__main__':
    main()
