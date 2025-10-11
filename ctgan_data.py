import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Check for GPU and set device
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("✓ CUDA detected. Using GPU for training.")
else:
    print("⚠ No CUDA device found. Using CPU.")

# Load the data
print("Loading data from 'augmented_with_signals.csv'...")
df = pd.read_csv('augmented_with_signals.csv')

# Drop non-numeric columns before training
df_numeric = df.select_dtypes(include='number')

# Create metadata
print("Detecting metadata from the dataframe...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_numeric)

# Save metadata for reproducibility
print("Metadata saved to metadata.json")
metadata.save_to_json(filepath='metadata.json')

# Initialize and train CTGAN with GPU and verbose progress
print("Initializing CTGANSynthesizer...")
synthesizer = CTGANSynthesizer(
    metadata,
    batch_size=1000,  # Increased batch size
    epochs=500,  # Increased epochs for the larger model
    embedding_dim=256,  # Increased model complexity
    generator_dim=(512, 512),  # Increased model complexity
    discriminator_dim=(512, 512),  # Increased model complexity
    cuda=use_cuda,
    verbose=True  # This will print training progress
)

print("Training the CTGAN model...")
synthesizer.fit(df_numeric)

# Generate synthetic data to reach 10,000 rows
num_synthetic_samples = 10000 - len(df_numeric)
print(f"Generating {num_synthetic_samples} new synthetic samples...")
synthetic_data = synthesizer.sample(
    num_rows=num_synthetic_samples
)

# Combine original and synthetic data
print("Combining original and synthetic data...")
final_data = pd.concat([df_numeric, synthetic_data], ignore_index=True)

# Save the final dataset
final_data.to_csv('final_data.csv', index=False)

# Calculate and save correlation matrix
print("Calculating and saving the final correlation matrix...")
# Ensure only numeric columns are used for correlation
numeric_final_data = final_data.select_dtypes(include='number')
correlation_matrix = numeric_final_data.corr()
correlation_matrix.to_csv('correlation_matrix_final.csv')

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', center=0) # annot=False for large matrices
plt.title('Correlation Matrix - Final Data (10,000 rows)')
plt.tight_layout()
plt.savefig('correlation_matrix_final.png')
plt.close()

print(f"Original data size: {len(df)}")
print(f"Final data size: {len(final_data)}")
print("Data saved to final_data.csv")
print("Correlation matrix saved to correlation_matrix_final.csv and correlation_matrix_final.png")
print("\nProcessing complete!")