def text_parse(big_string):
    """
    接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    import re
    list_of_tokens = re.split(r'\W+', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list(dataset):
    """
    创建一个包含在所有文档中出现的不重复的词的列表。
    """
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)  # union of the two sets
    return list(vocab_set)


def words_to_vec(vocab_list, input_set):
    """
    获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def cross_validation(estimator, X, y, cv=3):
    """
    交叉验证
    """
    import random
    # 生成随机下标
    indexes = list(range(len(X)))
    random.shuffle(indexes)
    errors_rate = []
    errors_index = []

    def map_data(source, data):
        return [source[val] for val in data]

    for i in range(cv):
        train_index = [index for index in indexes if index % cv != i]
        test_index = [index for index in indexes if index % cv == i]
        estimator.fit(map_data(X, train_index), map_data(y, train_index))
        error_index = []
        for j in test_index:
            res = estimator.predict([X[j]])[0] == y[j]
            if not res:
                error_index.append(j)
        errors_rate.append(len(error_index) / len(test_index))
        errors_index.append(error_index)
    return errors_rate, errors_index


def main():
    words_list = []
    class_list = []

    import os
    for ham in os.listdir('data/ham'):
        with open('data/ham/' + ham, 'r', encoding='iso-8859-1') as f:
            words_list.append(text_parse(f.read()))
            class_list.append(1)

    for spam in os.listdir('data/spam'):
        with open('data/spam/' + spam, 'r', encoding='iso-8859-1') as f:
            words_list.append(text_parse(f.read()))
            class_list.append(0)

    # 创建词汇表
    vocab_list = create_vocab_list(words_list)
    # 创建特征向量
    vec_list = [words_to_vec(vocab_list, words) for words in words_list]

    from sklearn.naive_bayes import MultinomialNB
    errors_rate, errors_index = cross_validation(MultinomialNB(), vec_list, class_list, cv=5)
    print(errors_rate, errors_index)
    print('平均错误率：', sum(errors_rate) / len(errors_rate))


if __name__ == '__main__':
    main()
