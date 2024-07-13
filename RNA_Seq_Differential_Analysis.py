import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Generate synthetic RNA-Seq data
np.random.seed(42)
genes = [f'Gene_{i}' for i in range(1, 201)]
samples = [f'Sample_{i}' for i in range(1, 21)]
data = np.random.poisson(lam=20, size=(200, 20))

# Introduce differential expression for a subset of genes
data[0:10, 10:] += 10  # Differentially expressed genes in the last 10 samples

# Create DataFrame
df = pd.DataFrame(data, index=genes, columns=samples)

# Split data into two groups
group1 = df.iloc[:, :10]
group2 = df.iloc[:, 10:]

# Perform t-test
p_values = []
for gene in df.index:
    stat, p_value = ttest_ind(group1.loc[gene], group2.loc[gene])
    p_values.append(p_value)

df['p_value'] = p_values
df['-log10(p_value)'] = -np.log10(df['p_value'])
df['significant'] = df['p_value'] < 0.05

# Plot Volcano Plot
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x=df.index, y='-log10(p_value)', hue='significant', data=df, palette={True: 'red', False: 'blue'})
plt.axhline(y=-np.log10(0.05), color='green', linestyle='--')
plt.title('Volcano Plot')
plt.xlabel('Gene')
plt.ylabel('-log10(p_value)')

# Plot heatmap of differentially expressed genes
plt.subplot(1, 3, 2)
significant_genes = df[df['significant']].index
sns.heatmap(df.loc[significant_genes].iloc[:, :-3], cmap='coolwarm', yticklabels=significant_genes)
plt.title('Heatmap of Differentially Expressed Genes')

# Plot histogram of p-values
plt.subplot(1, 3, 3)
plt.hist(df['p_value'], bins=50, edgecolor='black', color='purple')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('Histogram of p-values')

plt.tight_layout()
plt.show()
