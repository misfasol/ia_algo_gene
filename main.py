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
    tempo_treinamento = 0
    iteracao = 0
    inicio = time()
    while True:
        iteracao += 1
        melhorou = False
        print(f"{iteracao = }")
        features_agora = melhores_features.copy()
        for feature_i in range(treino_dados.shape[1]):
            if feature_i in melhores_features:
                continue
            novas_features = features_agora.copy()
            novas_features.append(feature_i)
            clf = DecisionTreeClassifier()
            adicionado = treino_dados.iloc[:, novas_features]
            treino_inicio = time()
            # print(feature_i)
            clf.fit(adicionado, treino_labels)
            # print("terminou treino", feature_i)
            tempo_treino = time() - treino_inicio
            acuracia = clf.score(teste_dados.iloc[:, novas_features], teste_labels)
            # print("acuracia", feature_i, acuracia)
            if acuracia > melhor_acuracia:
                melhor_acuracia = acuracia
                melhores_features = novas_features
                tempo_treinamento = tempo_treino
                melhorou = True
        print(f"{iteracao = }")
        print(f"{melhor_acuracia = }")
        print(f"{melhores_features = }")
        if not melhorou:
            break
    tempo_total = time() - inicio
    # print(melhor_acuracia, melhores_features)
    print(f"{melhor_acuracia = }")
    porcentagem = len(melhores_features) / treino_dados.shape[1]
    print(f"{porcentagem = }")
    print(f"{tempo_treinamento = }")
    print(f"{tempo_total = }")

    print("Wrapper")


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
    # tree_baseline()

if __name__ == "__main__":
    main()
