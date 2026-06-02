import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


NUMERO_DE_CLUSTERS = 3 


df = pd.read_excel('../Database/RTVue_20221110_MLClass_Tratado.xlsx')
colunas_modelo = ['Age', 'Gender', 'Eye', 'C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
X = df[colunas_modelo]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=NUMERO_DE_CLUSTERS, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
df['PCA3'] = X_pca[:, 2]



# GERAÇÃO DO GRÁFICO 1: VISUALIZAÇÃO 2D
plt.figure(figsize=(10, 8))

sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette=sns.color_palette("tab10", NUMERO_DE_CLUSTERS),
    data=df, s=100, alpha=0.8
)

plt.title(f'Clusters K-Means em 2D (K={NUMERO_DE_CLUSTERS})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

caminho_2d = f'Grafics/kmeans_2D_k{NUMERO_DE_CLUSTERS}.png'
plt.savefig(caminho_2d, bbox_inches='tight', dpi=300)
plt.close()



# GERAÇÃO DO GRÁFICO 2: VISUALIZAÇÃO 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')


for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    ax.scatter(
        cluster_data['PCA1'], 
        cluster_data['PCA2'], 
        cluster_data['PCA3'], 
        label=f'Cluster {cluster_id}',
        s=80, alpha=0.8
    )

ax.set_title(f'Clusters K-Means em 3D (K={NUMERO_DE_CLUSTERS})')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend(title='Cluster')

caminho_3d = f'Grafics/kmeans_3D_k{NUMERO_DE_CLUSTERS}.png'
plt.savefig(caminho_3d, bbox_inches='tight', dpi=300)
plt.close()


print(f"\n[Sucesso] Algoritmo rodou agrupando os dados em {NUMERO_DE_CLUSTERS} clusters.")
print(f"-> Gráfico 2D gerado em: {caminho_2d}")
print(f"-> Gráfico 3D gerado em: {caminho_3d}")