from sklearn.tree import DecisionTreeClassifier

def main():
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)
    print(clf.predict([[2, 2]]))
    print("oi")

if __name__ == "__main__":
    main()
