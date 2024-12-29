from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import pandas as pd
import Data_Plots as dp

def find_mean_values(data): # Define the function that calculates the mean values of the data
    return data.mean()

def find_standard_deviation(data): # Define the function that calculates the standard deviation of the data
    return data.std()

def pca_process(data, percentage): # Define the function that performs PCA on the data
    z_data = stats.zscore(data)
    covariance_matrix = z_data.cov() # Calculate the covariance matrix of the data
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix) # Calculate the eigenvalues and eigenvectors of the covariance matrix
    idx = eigenvalues.argsort()[::-1] # Sort the eigenvalues in descending order
    eigenvalues = eigenvalues[idx] # Rearrange the eigenvalues
    eigenvectors = eigenvectors[:, idx] # Rearrange the eigenvectors
    explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues) # Calculate the explained variance
    print("Explained variance:\n", explained_variance) # Print the explained variance

    n_components = np.argmax(explained_variance > percentage) + 1 # Find the number of components that explain at least 95% of the variance
    pca = PCA(n_components = n_components) # Create a PCA object with the number of components
    pca.fit(z_data) # Fit the PCA object to the data
    transformed_data = pca.transform(z_data) # Transform the data using the PCA object
    transformed_data_df = pd.DataFrame(transformed_data, columns = ['PC{}'.format(i + 1) for i in range(n_components)]) # Convert the transformed data to a DataFrame
    print(transformed_data_df.head(10)) # Print the transformed data
    dp.scatter_plot(10, 10, transformed_data_df['PC1'], transformed_data_df['PC2'],'PC1', 'PC2', 'Scatter plot of principal components') # Create a scatter plot of the transformed data
