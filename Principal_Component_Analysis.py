import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Generate synthetic gene expression data (base data is needed for expression)
np.random.seed(42)
genes = [f'Gene_{i}' for i in range(1, 101)]
samples = [f'Sample_{i}' for i in range(1, 21)]
data = np.random.normal(loc=10, scale=2, size=(100, 20))


df = pd.DataFrame(data, index=genes, columns=samples)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.T)


pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=samples)

# Plotting the PCA results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title('PCA of Gene Expression Data')


explained_variance = pca.explained_variance_ratio_

plt.subplot(1, 2, 2)
plt.bar(['PC1', 'PC2'], explained_variance)
plt.title('Explained Variance')
plt.ylabel('Proportion of Variance')

plt.tight_layout()
plt.show()
