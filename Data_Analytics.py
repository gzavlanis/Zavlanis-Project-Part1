from sklearn.decomposition import PCA
from scipy import stats

def find_mean_values(data): # Define the function that calculates the mean values of the data
    return data.mean()

def find_standard_deviation(data): # Define the function that calculates the standard deviation of the data
    return data.std()

def z_score_normalization(data): # Define the function that normalizes the data using z-score normalization
    return stats.zscore(data)

def
