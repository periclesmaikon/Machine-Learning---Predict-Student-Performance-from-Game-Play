import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import gc # Garbage Collector para limpar a memória

#CARREGAMENTO EFICIENTE DE DADOS
print("Carregando os dados...")

#Definir os tipos de dados para economizar memória RAM!
dtypes = {
    'session_id': 'int64',
    'elapsed_time': 'int32',
    'event_name': 'category',
    'level': 'uint8',
    'level_group': 'category'
}

#Lendo apenas as colunas para o modelo simples
use_cols = ['session_id', 'elapsed_time', 'event_name', 'level', 'level_group']

caminho_base = '/kaggle/input/predict-student-performance-from-game-play/'

train = pd.read_csv(caminho_base + 'train.csv', dtype=dtypes, usecols=use_cols)
targets = pd.read_csv(caminho_base + 'train_labels.csv')

#O target vem no formato "20090312431273200_q1". Separando o ID da Sessão e a Questão.
targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))
targets['q'] = targets.session_id.apply(lambda x: int(x.split('_q')[1]))

#ENGENHARIA DE FEATURES
print("Criando variáveis...")

def feature_engineer(df):
    """
    Função que transforma as linhas de eventos em 1 linha por sessão + grupo de níveis.
    """
    #Extração de estatísticas básicas: o tempo máximo gasto e quantas ações foram feitas
    agregacoes = {
        'elapsed_time': ['max', 'mean'],
        'level': ['max']
    }
    
    #Agrupamento por sessão e pelo grupo de níveis
    df_agrupado = df.groupby(['session_id', 'level_group']).agg(agregacoes)
    
    df_agrupado.columns = ['_'.join(col).strip() for col in df_agrupado.columns.values]
    df_agrupado = df_agrupado.reset_index()
    
    return df_agrupado

#Função no treino
train_features = feature_engineer(train)

del train
gc.collect()

#REPARAÇÃO PARA O TREINO
print("Preparando para treinar a Random Forest...")

#Definindo as variáveis que o modelo vai usar
features = [c for c in train_features.columns if c not in ['session_id', 'level_group']]

#Treinando 3 modelos, um para cada grupo de níveis, pois eles respondem a questões diferentes
grupos = ['0-4', '5-12', '13-22']
modelos = {}

for grupo in grupos:
    print(f"Treinando modelo para o grupo {grupo}...")
    
    #Filtro de dados do grupo atual
    df_grupo = train_features[train_features['level_group'] == grupo]
    
    #Descobrir quais questões pertencem a este grupo
    if grupo == '0-4':
        questoes = range(1, 4) #Questões 1, 2, 3
    elif grupo == '5-12':
        questoes = range(4, 14) #Questões 4 a 13
    else:
        questoes = range(14, 19) #Questões 14 a 18
        
    #Treino de um modelo para cada questão do grupo
    for q in questoes:
        #Filtro do target para a questão atual
        target_q = targets[targets['q'] == q]
        
        #Junção da features com o target
        dados_treino = df_grupo.merge(target_q, left_on='session_id', right_on='session', how='inner')
        
        X = dados_treino[features]
        y = dados_treino['correct']
        
        #Modelo Random Forest (n_estimators=100 é um bom padrão, max_depth limita o tamanho da árvore para eficiência)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        #Salvar o modelo treinado no dicionário
        modelos[f'{grupo}_q{q}'] = rf

print("Treinamento concluído!")

#SUBMISSÃO COM A API DO KAGGLE
print("Iniciando a inferência na API de tempo real...")

import sys
sys.path.append('/kaggle/input/predict-student-performance-from-game-play')

import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()

for (test_df, sample_submission) in iter_test:
    grp = test_df.level_group.iloc[0]
    test_df = feature_engineer(test_df)
    
    for q in sample_submission['session_id'].apply(lambda x: int(x.split('_q')[1])).unique():
        modelo_atual = modelos[f'{grp}_q{q}']
        
        X_test = test_df[features]
        
        #A API pede apenas 0 (erro) ou 1 (acerto)
        predicao = modelo_atual.predict(X_test)[0]
        
        #Arquivo de submissão
        mask = sample_submission['session_id'].str.contains(f'_q{q}')
        sample_submission.loc[mask, 'correct'] = int(predicao)
    
    #Envio da previsão desta etapa para a API
    env.predict(sample_submission)

print("Submissão gerada com sucesso!")
