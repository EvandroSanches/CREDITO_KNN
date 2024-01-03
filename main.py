from pyod.models.knn import KNN
from pandasgui import show
import pandas as pd


#Carregando base de dados
dados = pd.read_csv('credit_data.csv')

#Tratando valores nulos e idades negativas
dados = dados.dropna()
dados.age[dados.age < 0] = dados.age[dados.age < 0] * -1

#Criando e Treinando modelo
detector = KNN()
detector.fit(dados)

#Atribuindo classificações a variavel previsores
previsores = detector.labels_

outliers = []

#Pegando indice dos valores classificados como outlier
for i in range(len(previsores)):
    if previsores[i] == 1:
        outliers.append(i)

dados = dados.iloc[outliers, :]

#Imprimindo dados
show(dados)







