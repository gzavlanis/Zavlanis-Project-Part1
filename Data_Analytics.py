from unittest.mock import inplace

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import pandas as pd
import Data_Plots as dp

def find_mean_values(data): # Define the function that calculates the mean values of the data
    return data.mean()

def find_standard_deviation(data): # Define the function that calculates the standard deviation of the data
    return data.std()

def pca_process(data, percentage, dataset_name): # Define the function that performs PCA on the data
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
    pc_values = np.arange(pca.n_components_) + 1 # Create an array of the principal component values

    dp.scree_plot(pc_values, pca.explained_variance_ratio_, 'Principal Component', 'Explained Variance', 'Scree plot of principal components', f'./Plots/{dataset_name}_Scree.png', 'blue') # Create a scree plot of the principal components
    dp.scatter_plot(transformed_data_df['PC1'], transformed_data_df['PC2'],'PC1', 'PC2', 'Scatter plot of principal components', f'./Plots/{dataset_name}.png') # Create a scatter plot of the transformed data

def hypothesis_testing(group1, group2):
    return stats.ttest_ind(group1, group2) # Perform a t-test on the two groups

def create_gradient_descent_dataset(original_data):
    original_data['Exam'] = original_data[['Final Exam', 'Repeat Exam']].max(axis=1)  # Create a new column 'Exam' based on the final and repeat exam grades
    original_data.drop(['Final Exam', 'Repeat Exam'], axis = 1, inplace = True)  # Drop the final and repeat exam columns
    X = original_data.drop(columns = ['Exam'])  # Get the features
    X['Mean_Homework'] = X.iloc[:, :3].mean(axis = 1)
    X['Mean_Compulsory_Activities'] = X.iloc[:, 4:11].mean(axis = 1)
    X['Mean_Optional_Activities'] = X.iloc[:, 12:21].mean(axis = 1)
    X.drop([
        'Homework 1',
        'Homework 2',
        'Homework 3',
        'Homework 4',
        'Compulsory Activity 1',
        'Compulsory Activity 2',
        'Compulsory Activity 3',
        'Compulsory Activity 4',
        'Compulsory Activity 5',
        'Compulsory Activity 6',
        'Compulsory Activity 7',
        'Compulsory Activity 8',
        'Optional Activity 1',
        'Optional Activity 2',
        'Optional Activity 3',
        'Optional Activity 4',
        'Optional Activity 5',
        'Optional Activity 6',
        'Optional Activity 7',
        'Optional Activity 8',
        'Optional Activity 9',
        'Optional Activity 10'
    ], axis = 1, inplace = True)
    y = original_data['Exam']  # Get the target
    scaler = StandardScaler()  # Create a StandardScaler object
    X_scaled = scaler.fit_transform(X)  # Scale the features
    return X, X_scaled, y

def gradient_descent_process(X, y):
    model = SGDRegressor(max_iter = 1000, tol = 1e-3, learning_rate = 'constant', eta0 = 0.01) # Create a SGDRegressor object
    model.fit(X, y) # Fit the model to the data
    predictions = model.predict(X) # Make predictions
    mse = mean_squared_error(y, predictions) # Calculate the mean squared error

    print("\nOptimized Parameters (Theta):", model.coef_)
    print("\nBias Term (Intercept):", model.intercept_)
    print("\nMean Squared Error:", mse)

    dp.plot_of_actual_and_predicted(X[:, 0], y, predictions, 'X (scaled)', 'Exam', 'Actual vs Predicted Exam Marks', './Plots/Actual_vs_Predicted_Exam_Marks.png') # Create a plot of the actual and predicted exam marks

def predict(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def custom_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = predict(X, theta)
        theta -= (alpha / m) * np.dot(X.T, (predictions - y))
        cost_history[i] = compute_cost(X, y, theta)
    return theta, cost_history
