import random
from sklearn.tree import DecisionTreeClassifier
from time import time

class AlgoritmoGenetico:

    
    def __init__(self, treino_dados, treino_labels, teste_dados, teste_labels, 
                 tamanho_populacao=20, geracoes=50, taxa_mutacao=0.1):
        self.treino_dados = treino_dados
        self.treino_labels = treino_labels
        self.teste_dados = teste_dados
        self.teste_labels = teste_labels
        self.num_features = treino_dados.shape[1]
        
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.historico_fitness = []
        self.historico_num_features = []
    
    def criar_individuo(self):
        # Garante que pelo menos 5 features sejam selecionadas inicialmente
        cromossomo = [0] * self.num_features
        num_features_iniciais = random.randint(5, min(50, self.num_features))
        
        features_selecionadas = random.sample(range(self.num_features), num_features_iniciais)
        for feature in features_selecionadas:
            cromossomo[feature] = 1
            
        return cromossomo
    
    def get_features_selecionadas(self, cromossomo):
        # Retorna lista de índices das features selecionadas
        return [i for i, gene in enumerate(cromossomo) if gene == 1]
    
    def fitness(self, cromossomo):
        # Função fitness: acurácia do modelo - penalização por muitas features
        features_selecionadas = self.get_features_selecionadas(cromossomo)
        
        if len(features_selecionadas) == 0:
            return 0.0
        
        # Treina modelo apenas com features selecionadas
        X_treino = self.treino_dados.iloc[:, features_selecionadas]
        X_teste = self.teste_dados.iloc[:, features_selecionadas]
        
        clf = DecisionTreeClassifier(random_state=42, max_depth=10)  # Limita profundidade para acelerar
        clf.fit(X_treino, self.treino_labels)
        acuracia = clf.score(X_teste, self.teste_labels)
        
        # Penalização suave por usar muitas features
        penalizacao = (len(features_selecionadas) / self.num_features) * 0.05
        fitness = acuracia - penalizacao
        
        return max(0.0, fitness)
    
    def selecao_roleta(self, populacao, fitness_scores):
        # Seleção por roleta viciada
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(populacao)
        
        r = random.uniform(0, total_fitness)
        soma_parcial = 0
        
        for i, fitness in enumerate(fitness_scores):
            soma_parcial += fitness
            if soma_parcial >= r:
                return populacao[i].copy()
        
        return populacao[-1].copy()
    
    def crossover_um_ponto(self, pai1, pai2):
        # Crossover de um ponto
        ponto = random.randint(1, len(pai1) - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
        return filho1, filho2
    
    def mutacao(self, cromossomo):
        # Mutação bit-flip com garantia de pelo menos uma feature
        cromossomo_mutado = cromossomo.copy()
        
        for i in range(len(cromossomo_mutado)):
            if random.random() < self.taxa_mutacao:
                cromossomo_mutado[i] = 1 - cromossomo_mutado[i]
        
        # Garante que pelo menos 3 features sejam selecionadas
        if sum(cromossomo_mutado) < 3:
            features_para_ativar = random.sample(range(len(cromossomo_mutado)), 3)
            for f in features_para_ativar:
                cromossomo_mutado[f] = 1
                
        return cromossomo_mutado
    
    def evoluir(self):

        inicio_total = time()
        
        # População inicial
        populacao = [self.criar_individuo() for _ in range(self.tamanho_populacao)]
        
        melhor_individuo_global = None
        melhor_fitness_global = -1
        
        for geracao in range(self.geracoes):
            # Avaliação
            fitness_scores = [self.fitness(ind) for ind in populacao]
            
            # Estatísticas da geração
            melhor_fitness = max(fitness_scores)
            fitness_medio = sum(fitness_scores) / len(fitness_scores)
            melhor_individuo = populacao[fitness_scores.index(melhor_fitness)]
            num_features_melhor = sum(melhor_individuo)
            
            self.historico_fitness.append(melhor_fitness)
            self.historico_num_features.append(num_features_melhor)
            
            # Atualiza melhor global
            if melhor_fitness > melhor_fitness_global:
                melhor_fitness_global = melhor_fitness
                melhor_individuo_global = melhor_individuo.copy()
            
            # Mostra progresso
            if geracao % 5 == 0 or geracao == self.geracoes - 1:
                print(f"Geração {geracao:2d}: Fitness={melhor_fitness:.4f}, "
                      f"Features={num_features_melhor:3d}, Média={fitness_medio:.4f}")
            
            # Critério de parada
            if geracao >= 15:
                ultimos_5 = self.historico_fitness[-5:]
                if all(abs(f - melhor_fitness) < 0.001 for f in ultimos_5):
                    print(f"Convergência detectada na geração {geracao}")
                    break
            
            # Nova geração
            nova_populacao = []
            
            # Elitismo
            indices_fitness = sorted(range(len(fitness_scores)), 
                                   key=lambda i: fitness_scores[i], reverse=True)
            nova_populacao.append(populacao[indices_fitness[0]].copy())
            if len(populacao) > 1:
                nova_populacao.append(populacao[indices_fitness[1]].copy()) # preserva os 2 melhores
            
            # Preenche resto da população
            while len(nova_populacao) < self.tamanho_populacao:
                # Seleção
                pai1 = self.selecao_roleta(populacao, fitness_scores)
                pai2 = self.selecao_roleta(populacao, fitness_scores)
                
                # Crossover
                if random.random() < 0.7:  # 70% de chance de crossover
                    filho1, filho2 = self.crossover_um_ponto(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                
                # Mutação
                filho1 = self.mutacao(filho1)
                filho2 = self.mutacao(filho2)
                
                nova_populacao.extend([filho1, filho2])
            
            populacao = nova_populacao[:self.tamanho_populacao]
        
        tempo_total = time() - inicio_total
        
        # Avaliação final detalhada
        features_finais = self.get_features_selecionadas(melhor_individuo_global)
        X_treino_final = self.treino_dados.iloc[:, features_finais]
        X_teste_final = self.teste_dados.iloc[:, features_finais]
        
        inicio_treino = time()
        clf_final = DecisionTreeClassifier(random_state=42)
        clf_final.fit(X_treino_final, self.treino_labels)
        tempo_treino = time() - inicio_treino
        
        acuracia_final = clf_final.score(X_teste_final, self.teste_labels)
        
        print(f"Melhor fitness: {melhor_fitness_global:.4f}")
        print(f"Acurácia final: {acuracia_final:.4f}")
        print(f"Features selecionadas: {len(features_finais)}")
        print(f"Porcentagem de features: {len(features_finais)/self.num_features:.2%}")
        print(f"Tempo de treino: {tempo_treino:.4f}s")
        print(f"Tempo total: {tempo_total:.2f}s")
        print(f"Primeiras features: {features_finais[:15]}...")
        

        return melhor_individuo_global, {
            'fitness': melhor_fitness_global,
            'acuracia': acuracia_final,
            'features_selecionadas': features_finais,
            'num_features': len(features_finais),
            'porcentagem_features': len(features_finais) / self.num_features,
            'tempo_treino': tempo_treino,
            'tempo_total': tempo_total,
            'historico_fitness': self.historico_fitness,
            'historico_num_features': self.historico_num_features
        }


def algoritmo_genetico(treino_dados, treino_labels, teste_dados, teste_labels):

    ag = AlgoritmoGenetico(
        treino_dados=treino_dados,
        treino_labels=treino_labels,
        teste_dados=teste_dados,
        teste_labels=teste_labels,
        tamanho_populacao=30,  # População moderada
        geracoes=25,           # Poucas gerações para ser rápido
        taxa_mutacao=0.1       # Mutação moderada
    )
    
    resultado = ag.evoluir()
    return resultado
