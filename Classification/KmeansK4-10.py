import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Carregando a base de dados
df = pd.read_excel('../Database/RTVue_20221110_MLClass_Tratado.xlsx')
colunas_modelo = ['Age', 'Gender', 'Eye', 'C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
X = df[colunas_modelo]

# 2. Normalização dos Dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Aplicando PCA (Feito ANTES do loop por questão de performance)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
df['PCA3'] = X_pca[:, 2]

# 4. Loop Automatizado para K variando de 4 até 10
# A função range(4, 11) inclui o 4 e vai até o 10
for k in range(4, 11):
    print(f"\nProcessando agrupamento para K={k}...")

    # Executa o K-Means para o valor de K atual da repetição
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # ==========================================
    # GERAÇÃO E SALVAMENTO DO GRÁFICO 2D
    # ==========================================
    plt.figure(figsize=(10, 8))

    # Usamos a paleta 'husl' porque ela gera cores distintas perfeitamente
    # independente do tamanho do K.
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        palette=sns.color_palette("husl", k),
        data=df, s=100, alpha=0.8
    )

    plt.title(f'Clusters K-Means em 2D (K={k})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # Salvando na pasta Grafics_2D conforme a sua estrutura
    caminho_2d = f'Grafics_2D/kmeans_2D_k{k}.png'
    plt.savefig(caminho_2d, bbox_inches='tight', dpi=300)
    plt.close()

    # ==========================================
    # GERAÇÃO E SALVAMENTO DO GRÁFICO 3D
    # ==========================================
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Puxa a mesma paleta de cores 'husl' gerada para o seaborn para manter o padrão
    cores = sns.color_palette("husl", k)

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['PCA1'], 
            cluster_data['PCA2'], 
            cluster_data['PCA3'], 
            color=cores[cluster_id],
            label=f'Cluster {cluster_id}',
            s=80, alpha=0.8
        )

    ax.set_title(f'Clusters K-Means em 3D (K={k})')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.legend(title='Cluster')

    # Salvando na pasta Graficos_3D conforme a sua estrutura
    caminho_3d = f'Graficos_3D/kmeans_3D_k{k}.png'
    plt.savefig(caminho_3d, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"-> Salvo: {caminho_2d}")
    print(f"-> Salvo: {caminho_3d}")

print("\n[Sucesso] Todos os 14 gráficos foram gerados e organizados nas respectivas pastas!")