import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import os
from sklearn.linear_model import LogisticRegression
#from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import libs.metrics_ml as m_plt


class modelSetting():
    
    def __init__(self, base, target, split_sample=None, cols_combinations=None, test_size=None, models=None, mlflow_name_experiment=None):
        
        self.base = base # Dataset
        self.target = target # Variável resposta
        self.split_sample = split_sample # Variável para particionar o dataset
        self.cols_combinations = cols_combinations # Variável com as combinações de features
        self.test_size = test_size # Variável para definir a divisão de dados de treino e teste
        self.models = models
        self.models_to_run = []
        self.mlflow_name_experiment = mlflow_name_experiment
        
        # Instanciando os modelos
        for name_model in self.models:
            if name_model == 'LogisticRegression':
                self.models_to_run.append([LogisticRegression(random_state=42), "LogisticRegression", "sklearn"])
            elif name_model == 'XGBClassifier':
                self.models_to_run.append([XGBClassifier(random_state=42), "XGBClassifier", "XGBoost"])
            elif name_model == 'LGBMClassifier':
                self.models_to_run.append([LGBMClassifier(random_state=42), "LGBMClassifier", "LGBoost"])
            elif name_model == 'RandomForestClassifier':
                self.models_to_run.append([RandomForestClassifier(random_state=42), "RandomForestClassifier", "sklearn"])
            elif name_model == 'AdaBoostClassifier':
                self.models_to_run.append([AdaBoostClassifier(random_state=42), "AdaBoostClassifier", "sklearn"])
            elif name_model == 'GradientBoostingClassifier':
                self.models_to_run.append([GradientBoostingClassifier(random_state=42), "GradientBoostingClassifier", "sklearn"])
            elif name_model == 'LinearDiscriminantAnalysis':
                self.models_to_run.append([LinearDiscriminantAnalysis(), "LinearDiscriminantAnalysis", "sklearn"])
            elif name_model == 'GaussianNB':
                self.models_to_run.append([GaussianNB(), "GaussianNB", "sklearn"])
            elif name_model == 'DecisionTreeClassifier':
                self.models_to_run.append([DecisionTreeClassifier(random_state=42), "DecisionTreeClassifier", "sklearn"])
            elif name_model == 'KNeighborsClassifier':
                self.models_to_run.append([KNeighborsClassifier(), "KNeighborsClassifier", "sklearn"])
    
    def execute(self):
        
        name_experiment = self.mlflow_name_experiment
        experiment_id = mlflow.set_experiment(name_experiment)
        
        if self.split_sample:
            base_reject, self.base = train_test_split(self.base, test_size=self.split_sample, random_state=92)
            self.base = self.base.reset_index(drop=True)

        clf_plt = m_plt.classificationsPlotting()

        if not self.cols_combinations:

            dataset = self.base
            X = dataset.drop([self.target],axis=1)
            y = dataset[self.target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=92)

            dictionary_columns = {'columns': list(X.columns), 'target': self.target}

            for model, name, type_ml_package in self.models_to_run:
                
                if type_ml_package == "XGBoost":
                    mlflow.xgboost.autolog()
                
                with mlflow.start_run(run_name=name):
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:,1]

                    confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred)
                    metrics = clf_plt.getClassificationMetrics(y_test, y_pred)

                    TP = confusion_matrix.iloc[0,0]
                    FP = confusion_matrix.iloc[1,1]
                    TN = confusion_matrix.iloc[0,1]
                    FN = confusion_matrix.iloc[1,0]
                    accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
                    recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
                    specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
                    precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
                    f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
                    auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
                    kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)
                    
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("TP", TP)
                    mlflow.log_metric("FP", FP)
                    mlflow.log_metric("TN", TN)
                    mlflow.log_metric("FN", FN)
                    mlflow.log_metric("Recall", recall)
                    mlflow.log_metric("Specificity", specificity)
                    mlflow.log_metric("Precision", precision)
                    mlflow.log_metric("F1", f1)
                    mlflow.log_metric("AUC", auc)
                    mlflow.log_metric("Kappa", kappa)
                    mlflow.log_metric("Nº columns", X.shape[1])
                    mlflow.log_metric("X train", X_train.shape[0])
                    mlflow.log_metric("X test", X_test.shape[0])

                    mlflow.log_param("Features", X_train.columns)
                    mlflow.log_param("Target", self.target)

                    #if type_ml_package != "XGBoost":
                    #dict_parameters = model.get_params()
                    #for param, value in dict_parameters.items():
                    #    mlflow.log_param(param, value)
                    if type_ml_package != "XGBoost": 
                        mlflow.log_params(model.get_params())

                    #mlflow.log_dict(dictionary_columns, "columns.txt")
                

        else:
                
            for combination in self.cols_combinations:
                
                dataset = self.base
                X = dataset[combination]
                y = dataset[self.target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=92)

                dictionary_columns = {'columns': list(X.columns), 'target': self.target}
                
                for model, name, type_ml_package in self.models_to_run:
                    
                    if type_ml_package == "XGBoost":
                        mlflow.xgboost.autolog()

                    with mlflow.start_run(run_name=name):

                        model.fit(X_train, y_train)
                        y_pred = model.predict_proba(X_test)[:,1]

                        confusion_matrix = clf_plt.confusionMatrix(y_test, y_pred)
                        metrics = clf_plt.getClassificationMetrics(y_test, y_pred)

                        TP = confusion_matrix.iloc[0,0]
                        FP = confusion_matrix.iloc[1,1]
                        TN = confusion_matrix.iloc[0,1]
                        FN = confusion_matrix.iloc[1,0]
                        accuracy = round(metrics[metrics.index=='Accuracy']['Metrics'][0] * 100, 2) 
                        recall = round(metrics[metrics.index=='Recall']['Metrics'][0] * 100, 2)
                        specificity = round(metrics[metrics.index=='Specificity']['Metrics'][0] * 100, 2)
                        precision = round(metrics[metrics.index=='Precision']['Metrics'][0] * 100, 2)
                        f1 = round(metrics[metrics.index=='F1']['Metrics'][0] * 100, 2)
                        auc = round(metrics[metrics.index=='ROC AUC']['Metrics'][0] * 100, 2)
                        kappa = round(metrics[metrics.index=='Kappa']['Metrics'][0] * 100, 2)
                        
                        mlflow.log_metric("Accuracy", accuracy)
                        mlflow.log_metric("TP", TP)
                        mlflow.log_metric("FP", FP)
                        mlflow.log_metric("TN", TN)
                        mlflow.log_metric("FN", FN)
                        mlflow.log_metric("Recall", recall)
                        mlflow.log_metric("Specificity", specificity)
                        mlflow.log_metric("Precision", precision)
                        mlflow.log_metric("F1", f1)
                        mlflow.log_metric("AUC", auc)
                        mlflow.log_metric("Kappa", kappa)
                        mlflow.log_metric("Nº columns", X.shape[1])
                        mlflow.log_metric("X train", X_train.shape[0])
                        mlflow.log_metric("X test", X_test.shape[0])

                        #if type_ml_package != "XGBoost":
                        #dict_parameters = model.get_params()
                        #for param, value in dict_parameters.items():
                        #    mlflow.log_param(param, value)
                        if type_ml_package != "XGBoost": 
                            mlflow.log_params(model.get_params())
                        
                        mlflow.log_dict(dictionary_columns, "columns.txt")
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        