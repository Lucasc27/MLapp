import itertools as tools
import pandas as pd

class featureCombinations():
    
    def __init__(self, base, target, n_min=None, n_max=None, black_list=None, white_list=None):
        
        self.base = base # Dataset
        self.target = target # Variável resposta
        self.n_min = n_min # Número mínimo de combinações
        self.n_max = n_max # Número máximo de combinações
        self.black_list = black_list # Variáveis para excluir
        self.white_list = white_list # Variáveis constantes
        self.batch_combinations = [] # Variável para guardar tamanho das listas de combinações
        self.total_combination = [] # Variável para guardar número total de combinações
        
    def toCombine(self):

        # Removendo a coluna target
        self.cols = list(self.base.columns)
        self.cols.remove(self.target)

        # Removendo as variáveis da black list
        if self.black_list != None:
            for r in self.black_list:
                self.cols.remove(r)

        # Removendo as variáveis da white list
        if self.white_list != None:
            for r in self.white_list:
                self.cols.remove(r)

        # Validando os ranges min e max do loop de combinações
        self.n_min = 2 if not self.n_min else self.n_min
        self.n_max = len(self.cols) if not self.n_max else self.n_max

        # Loop de combinações <--------------------------------------------------------------------------------
        self.combinations_features = []
        for n in range(self.n_min, self.n_max+1):
            for x in tools.combinations(self.cols, n):
                self.combinations_features.append(list(x))
        
        # Incluindo as variáveis  da white list em todas as combinações
        if self.white_list != None:
            for comb in self.combinations_features:
                for x in self.white_list:
                    comb.append(x)
        #------------------------------------------------------------------------------------------------------
        
        # Contagem de combinações batch e total
        lista_contagem = []
        for x in range(0, len(self.combinations_features)):
            lista_contagem.append(len(self.combinations_features[x]))
            
        self.total_combination = len(self.combinations_features)
        self.batch_combinations = pd.DataFrame(lista_contagem, columns=['Qnt_var']).value_counts()
        
        # Retornando os resultados -----------------------------------------------------------------------------
        return self.combinations_features