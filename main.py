from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from time import time

treino = pd.read_csv("dados/mnist_train.csv")
# print(treino.head())
treino_labels = treino["label"]
# print(treino_labels.head())
treino_dados = treino.drop("label", axis=1)
# print(treino_dados.head())

teste = pd.read_csv("dados/mnist_test.csv")
# print(teste.head())
teste_labels = teste["label"]
# print(teste_labels)
teste_dados = teste.drop("label", axis=1)
# print(teste_dados)

def tree_ga():
    pass

def tree_wrapper():
    pass

def tree_baseline():
    clf = DecisionTreeClassifier(random_state=0)
    a = time()
    clf.fit(treino_dados, treino_labels)
    tempo_treino = time() - a
    acuracia = clf.score(teste_dados, teste_labels) * 100
    print("Baseline")
    print(f"{acuracia = }")
    print(f"porcentagem_features = 100")
    print(f"{tempo_treino = }")
    print(f"tempo_busca_features = 0")

def main():
    tree_ga()
    tree_wrapper()
    tree_baseline()

if __name__ == "__main__":
    main()
