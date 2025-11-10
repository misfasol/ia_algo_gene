from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dados_treino = pd.read_csv("dados/mnist_train.csv")
dados_teste = pd.read_csv("dados/mnist_test.csv")
print(dados_treino.head())
print(dados_teste.head())

def tree_ga():
    pass

def tree_wrapper():
    pass

def tree_baseline():
    clf = DecisionTreeClassifier()

def main():
    tree_ga()
    tree_wrapper()
    tree_baseline()

if __name__ == "__main__":
    main()
