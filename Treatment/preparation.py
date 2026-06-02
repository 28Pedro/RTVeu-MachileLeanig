import pandas as pd

df = pd.read_excel('../Database/RTVue_20221110_MLClass.xlsx')


colunas_interesse = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

print("--- Nulos ANTES do preenchimento ---")
print(df[colunas_interesse].isnull().sum())

df[colunas_interesse] = df[colunas_interesse].fillna(df[colunas_interesse].mean())

print("\n--- Nulos DEPOIS do preenchimento ---")
print(df[colunas_interesse].isnull().sum())

df = df.drop(columns=['Index', 'pID'], errors='ignore')

df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['Eye'] = df['Eye'].map({'OS': 0, 'OD': 1})


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print("\n--- Prévia das Primeiras Linhas da Base ---")
print(df.head())

caminho_saida = '../Database/RTVue_20221110_MLClass_Tratado.xlsx'
df.to_excel(caminho_saida, index=False)