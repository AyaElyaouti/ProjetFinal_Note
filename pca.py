import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():

    # 1️⃣ Charger le dataset
    df = pd.read_csv("/Users/ayaelyaouti/Documents/ProjetFinalTP/ProjetFinal_Note/city_lifestyle_dataset.csv")

    # 2️⃣ Supprimer les colonnes non numériques
    X = df.drop(columns=["city_name", "country"])

    # 3️⃣ Standardiser les données (TRÈS IMPORTANT pour PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4️⃣ Appliquer PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 5️⃣ Visualisation
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
    plt.title("Projection 2D des villes avec PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # 6️⃣ Exporter les données 2D
    np.savetxt("X_pca_2D.csv", X_pca, delimiter=",")

    # 7️⃣ Observation simple
    print("Variance expliquée :", pca.explained_variance_ratio_)

if __name__ == "__main__":
    main()