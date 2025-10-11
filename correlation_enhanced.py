import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('augmented_with_signals.csv')
# Drop non-numeric columns (e.g., dates, strings)
numeric_df = df.select_dtypes(include='number')
# Calculate correlation matrix only for numeric columns
correlation_matrix = numeric_df.corr()

# Create a figure with larger size for better visibility
plt.figure(figsize=(12, 10))

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title('Correlation Matrix Heatmap', fontsize=16, pad=20)
plt.tight_layout()

# Save the plot
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Correlation matrix saved as 'correlation_matrix.png'")

# Display the plot
plt.show()

# Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)