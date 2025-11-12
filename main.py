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
    melhores_features = []
    melhor_acuracia = 0
    iteracao = 0
    inicio = time()
    while iteracao < 2:
        iteracao += 1
        melhorou = False
        melhor_agora = 0
        print(f"{iteracao = }")
        inicio_iteracao = time()
        features_agora = melhores_features.copy()
        for feature_i in range(treino_dados.shape[1]):
            if feature_i in melhores_features:
                continue
            if feature_i % 100 == 0:
                print(feature_i, iteracao)
            features_agora.append(feature_i)
            clf = DecisionTreeClassifier(random_state=0)
            adicionado = treino_dados.iloc[:, features_agora]
            clf.fit(adicionado, treino_labels)
            acuracia = clf.score(teste_dados.iloc[:, features_agora], teste_labels)
            features_agora.pop()
            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhor_agora = feature_i
                melhorou = True
        if not melhorou:
            break
        else:
            melhores_features.append(melhor_agora)
        print(f"tempo_iteracao: {time() - inicio_iteracao}")
        print(f"{iteracao = }")
        print(f"{melhor_acuracia = }")
        print(f"{melhores_features = }")
            
    tempo_total = time() - inicio
    inicio = time()
    _ = DecisionTreeClassifier(random_state=0).fit(treino_dados.iloc[:, melhores_features], treino_labels)
    tempo_treinamento = time() - inicio
    porcentagem = len(melhores_features) / treino_dados.shape[1]

    print("Wrapper")
    print(f"acuracao = {melhor_acuracia}")
    print(f"porcentagem_features = {porcentagem}")
    print(f"tempo_treino = {tempo_treinamento}")
    print(f"tempo_busca_features = {tempo_total}")



def tree_baseline():
    clf = DecisionTreeClassifier(random_state=0)
    a = time()
    clf.fit(treino_dados, treino_labels)
    tempo_treino = time() - a
    acuracia = clf.score(teste_dados, teste_labels) * 100
    print("Baseline")
    print(f"acuracia = {acuracia}")
    print(f"porcentagem_features = 100")
    print(f"tempo_treino = {tempo_treino}")
    print(f"tempo_busca_features = 0")


def main():
    tree_ga()
    tree_wrapper()
    # tree_baseline()

if __name__ == "__main__":
    main()
