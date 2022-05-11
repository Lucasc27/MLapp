import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import metrics_ml as m_plt


class modelSetting():
    
    def __init__(self, base, target, split_sample=None, cols_combinations=None, test_size=None, auto_model=False):
        
        self.base = base # Dataset
        self.target = target # Variável resposta
        self.split_sample = split_sample # Variável para particionar o dataset
        self.cols_combinations = cols_combinations # Variável com as combinações de features
        #self.sample_base = None # Variável para guardar número de split_sample da base
        self.test_size = test_size # Variável para definir a divisão de dados de treino e teste
        self.list_shapes = [] # Variável que guarda os shapes da base e dos dados de treino
        self.list_models = [] # Variável que guarda a lista de modelos
        self.table_output = [] # Variável que contém os resultado finais
        self.auto_model = auto_model
        
        # Instanciando os modelos
        self.models_options = {
            
                1:[LogisticRegression(random_state=42), "LogisticRegression"],
                2:[CatBoostClassifier(random_state=42), "CatBoostClassifier"],
                3:[XGBClassifier(random_state=42), "XGBClassifier"],
                4:[LGBMClassifier(random_state=42), "LGBMClassifier"],
                5:[RandomForestClassifier(random_state=42), "RandomForestClassifier"],
                6:[AdaBoostClassifier(random_state=42), "AdaBoostClassifier"],
                7:[GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier"],
                8:[LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysis"],
                9:[GaussianNB(), "GaussianNB"],
                10:[DecisionTreeClassifier(random_state=42), "DecisionTreeClassifier"],
                11:[KNeighborsClassifier(), "KNeighborsClassifier"]
        }
        
        if self.auto_model:
            
            self.list_models = [[LogisticRegression(), "LogisticRegresion"]]
            
        print("* Para carregar algoritmos diferentes, salve o algoritmo em uma lista contendo o objeto e o nome dele. Ex: [[modelo:'name']]")
    
    def modelOpcions(self):
        
        print("---------------------")
        print("1 - LogisticRegresion")
        print("2 - CatBoostClassifier")
        print("3 - XGBClassifier")
        print("4 - LGBMClassifier")
        print("5 - RandomForestClassifier")
        print("6 - AdaBoostClassifier")
        print("7 - GradientBoostingClassifier")
        print("8 - LinearDiscriminantAnalysis")
        print("9 - GaussianNB")
        print("10 - DecisionTreeClassifier")
        print("11 - KNeighborsClassifier")
    
    def selectionModels(self, selection=None, replace=False):
        
        if not self.auto_model:
            
            if replace:
                self.list_models = []
        
            if isinstance(selection, list):
                
                for i in selection:
                    
                    if (i >= 1) and (i <= 11):
                        self.list_models.append(self.models_options[i])
                    else:
                        print("Seleção invalida")
                        
                print("Modelo(s) instanciado(s):",self.list_models)
                
            elif isinstance(selection, int):
                if (selection >= 1) and (selection <= 11):
                    self.list_models.append(self.models_options[selection])
                    print("Modelo(s) instanciado(s):",self.list_models)
                else:
                    print("Seleção invalida")

            else:

                print("Seleção invalida")
        else:
            
            print("Opção de auto model ativa! Para usar essa função desative o auto model")
    
    def inputModels(self, models):
        
        if not self.auto_model:
            self.list_models = models
            print("Modelo(s) instanciado(s):",self.list_models)
        else:
            print("Opção de auto model ativa! Para usar essa função desative o auto model")
        
    #def resetModels(self):
        
    #    if self.list_models:
    #        self.list_models = []    
    
    def showModels(self):
        
        if self.list_models:
            print(self.list_models)
        else:
            print("Não há modelos carregados")
    
    def execute(self):
        
        if self.list_models:
        
            if self.split_sample:
                if self.split_sample > 1:
                    self.split_sample = len(self.base) - 1
                else:
                    self.split_sample = self.split_sample
                #percent_base = self.split_sample / 100
                #self.sample_base = int(percent_base * len(self.base))
                base_reject, self.base = train_test_split(self.base, test_size=self.split_sample, random_state=92)
                self.base = self.base.reset_index(drop=True)

            if not self.cols_combinations:

                #dataset = self.base if not self.split_sample else self.base.sample(random_state=42, n=self.sample_base)
                dataset = self.base
                X = dataset.drop([self.target],axis=1)
                y = dataset[self.target]

                if self.test_size:
                    if self.test_size >= 1:
                        self.test_size = 0.3
                    else:
                        self.test_size = self.test_size

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=92)

                #self.list_shapes = []
                self.list_shapes = ({"X":X.shape,"y":y.shape,"X_train":X_train.shape,\
                                        "y_train":y_train.shape,"X_test":X_test.shape,"y_test":y_test.shape})
                
                self.table_output = pd.DataFrame(columns=['columns'],index=[0])
                self.table_output['columns'][0] = list(X.columns)
                
                for model, name in self.list_models:
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:,1]

                    # Definindo as classes positiva e negativa da variável target e criando uma lista com as categorias das classes.
                    labelPositive = 'Yes'
                    labelNegative = 'No'
                    labels = [labelPositive, labelNegative]

                    # Convertendo dados da variável target, dos dados de teste, para utilizar as labels especificadas.
                    testTargetLabels = [labelPositive if t == 1 else labelNegative for t in y_test]
                    testPredLabels   = [labelPositive if t >= 0.5 else labelNegative for t in y_pred]
                    
                    clf_plt = m_plt.classificationsPlotting()
                    cm = clf_plt.confusionMatrix(yTest = y_test, yPred = y_pred)
                    metrics = clf_plt.getClassificationMetrics(yTest = y_test, predProb = y_pred)
                    
                    self.table_output['('+name+')'] = 'ok'
                    self.table_output['TP_'+ name] = cm.iloc[0,0]
                    self.table_output['TN_'+ name] = cm.iloc[1,1]
                    self.table_output['FP_'+ name] = cm.iloc[0,1]
                    self.table_output['FN_'+ name] = cm.iloc[1,0]
                    self.table_output['accuracy_'+ name] = metrics[metrics.index=='Accuracy']['Metrics'][0]
                    self.table_output['Recall_'+ name] = metrics[metrics.index=='Recall (Sensitivity)']['Metrics'][0]
                    self.table_output['Specificity_'+ name] = metrics[metrics.index=='Specificity']['Metrics'][0]
                    self.table_output['Precision_'+ name] = metrics[metrics.index=='Precision']['Metrics'][0]
                    self.table_output['F1_'+ name] = metrics[metrics.index=='F1']['Metrics'][0]
                    self.table_output['ROC_AUC_'+ name] = metrics[metrics.index=='ROC AUC']['Metrics'][0]
                    self.table_output['Kappa_'+ name] = metrics[metrics.index=='Kappa']['Metrics'][0]
                    
                    self.table_output.to_csv('table_out_noCombination.csv')

                    # Calculando os scores de diferentes métricas, com base nas previsões geradas pelo modelo, para os dados de teste.
                    #self.result.append({1:clf_plt.getClassificationMetrics(yTrue = testTargetLabels, predProb = y_pred)})
                print("Execução finalizada com sucesso.")

            else:

                self.list_shapes = []
                self.table_output = pd.DataFrame(columns=['columns'],index=[list(range(0,len(self.cols_combinations)))])
                self.table_output['columns'] = None
                
                f = 0
                for model, name in self.list_models:
                    
                    self.table_output['('+name+')'] = 'ok'
                    self.table_output['TP_'+ name] = None
                    self.table_output['TN_'+ name] = None
                    self.table_output['FP_'+ name] = None
                    self.table_output['FN_'+ name] = None
                    self.table_output['accuracy_'+ name] = None
                    self.table_output['Recall_'+ name] = None
                    self.table_output['Specificity_'+ name] = None
                    self.table_output['Precision_'+ name] = None
                    self.table_output['F1_'+ name] = None
                    self.table_output['ROC_AUC_'+ name] = None
                    self.table_output['Kappa_'+ name] = None
                        
                    f = f + 1
                    
                for z,combination in enumerate(self.cols_combinations):
                    
                    #dataset = self.base if not self.split_sample else self.base.sample(random_state=42, n=self.sample_base)
                    dataset = self.base
                    X = dataset[combination]
                    y = dataset[self.target]

                    if self.test_size:
                        if self.test_size >= 1:
                            self.test_size = 0.3
                        else:
                            self.test_size = self.test_size

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=92)
                    
                    self.list_shapes = ({"X":X.shape,"y":y.shape,"X_train":X_train.shape,\
                                            "y_train":y_train.shape,"X_test":X_test.shape,"y_test":y_test.shape})
                    
                    for model, name in self.list_models:

                        self.table_output['columns'][z] = list(X.columns)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_test)[:,1]

                        # Definindo as classes positiva e negativa da variável target e criando uma lista com as categorias das classes.
                        #labelPositive = 'Yes'
                        #labelNegative = 'No'
                        #labels = [labelPositive, labelNegative]

                        # Convertendo dados da variável target, dos dados de teste, para utilizar as labels especificadas.
                        #testTargetLabels = [labelPositive if t == 1 else labelNegative for t in y_test]
                        #testPredLabels   = [labelPositive if t >= 0.5 else labelNegative for t in y_pred]
                        
                        clf_plt = m_plt.classificationsPlotting()
                        cm = clf_plt.confusionMatrix(yTest = y_test, yPred = y_pred)
                        metrics = clf_plt.getClassificationMetrics(yTest = y_test, predProb = y_pred)
                        
                        self.table_output['TP_'+ name][z] = cm.iloc[0,0]
                        self.table_output['TN_'+ name][z] = cm.iloc[1,1]
                        self.table_output['FP_'+ name][z] = cm.iloc[0,1]
                        self.table_output['FN_'+ name][z] = cm.iloc[1,0]
                        self.table_output['accuracy_'+ name][z] = metrics[metrics.index=='Accuracy']['Metrics'][0]
                        self.table_output['Recall_'+ name][z] = metrics[metrics.index=='Recall (Sensitivity)']['Metrics'][0]
                        self.table_output['Specificity_'+ name][z] = metrics[metrics.index=='Specificity']['Metrics'][0]
                        self.table_output['Precision_'+ name][z] = metrics[metrics.index=='Precision']['Metrics'][0]
                        self.table_output['F1_'+ name][z] = metrics[metrics.index=='F1']['Metrics'][0]
                        self.table_output['ROC_AUC_'+ name][z] = metrics[metrics.index=='ROC AUC']['Metrics'][0]
                        self.table_output['Kappa_'+ name][z] = metrics[metrics.index=='Kappa']['Metrics'][0]
                        
                        self.table_output.to_csv('table_out_withCombination.csv')
                
                print("Execução finalizada com sucesso.")
        
        
        else:
            print("Erro...carregue os modelos antes de usar essa função!")
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        