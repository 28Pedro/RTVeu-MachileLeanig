import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('../Database/RTVue_20221110_MLClass.xlsx')

print("--- Quantidade de Valores Nulos ---")
print(df.isnull().sum())


print("\n--- Resumo Estatístico ---")

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', 1000)


print(df.describe())

plot_coluns = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

sns.boxplot(data=df[plot_coluns])

plt.title('Identificação de Outliers por Coluna')
plt.xlabel('Colunas')
plt.ylabel('Valores')
plt.xticks(rotation=45)  


plt.savefig('boxplot_outliers.png', bbox_inches='tight')