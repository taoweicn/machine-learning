from decision_tree.decision_tree import DecisionTree


def main():
    with open('data/lenses.txt', 'r') as f:
        lenses = [line.strip().split('\t') for line in f.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    dt = DecisionTree()
    dt.fit(lenses)
    lenses_tree = dt.build_tree(lenses_labels)
    print(lenses_tree)
    dt.draw_tree(lenses_tree)


if __name__ == '__main__':
    main()
