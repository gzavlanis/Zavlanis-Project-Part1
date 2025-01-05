from Data_Processing import DataProcessing
import Data_Analytics as da
import numpy as np
import pandas as pd

def main():
    dataProcessing = DataProcessing() # Create an instance of the DataProcessing class
    exams_data, homework_data, compulsory_activities_data, optional_activities_data, original_data = dataProcessing.get_formatted_data() # Get the formatted data
    print("Exams data:\n", exams_data.head(5)) # Print the first 5 rows of the exams data
    print("\nHomework data:\n", homework_data.head(5)) # Print the first 5 rows of the homework data
    print("\nCompulsory activities data:\n", compulsory_activities_data.head(5)) # Print the first 5 rows of the compulsory activities data
    print("\nOptional activities data:\n", optional_activities_data.head(5)) # Print the first 5 rows of the optional activities data
    print("\nOriginal data:\n", original_data.head(5)) # Print the first 5 rows of the original data

    exams_mean_values = da.find_mean_values(exams_data) # Calculate the mean values of the exams data
    homework_mean_values = da.find_mean_values(homework_data) # Calculate the mean values of the homework data
    compulsory_activities_mean_values = da.find_mean_values(compulsory_activities_data) # Calculate the mean values of the compulsory activities data
    optional_activities_mean_values = da.find_mean_values(optional_activities_data) # Calculate the mean values of the optional activities data

    exams_standard_deviation = da.find_standard_deviation(exams_data) # Calculate the standard deviation of the exams data
    homework_standard_deviation = da.find_standard_deviation(homework_data) # Calculate the standard deviation of the homework data
    compulsory_activities_standard_deviation = da.find_standard_deviation(compulsory_activities_data) # Calculate the standard deviation of the compulsory activities data
    optional_activities_standard_deviation = da.find_standard_deviation(optional_activities_data) # Calculate the standard deviation of the optional activities data

    # Print the mean values and standard deviations
    print("\nExams mean values:\n", exams_mean_values)
    print("\nHomework mean values:\n", homework_mean_values)
    print("\nCompulsory activities mean values:\n", compulsory_activities_mean_values)
    print("\nOptional activities mean values:\n", optional_activities_mean_values)

    print("\nExams standard deviation:\n", exams_standard_deviation)
    print("\nHomework standard deviation:\n", homework_standard_deviation)
    print("\nCompulsory activities standard deviation:\n", compulsory_activities_standard_deviation)
    print("\nOptional activities standard deviation:\n", optional_activities_standard_deviation)

    da.pca_process(exams_data, 0.95, 'Exams_PCA') # Perform PCA on the exams data
    da.pca_process(homework_data, 0.95, 'Homework_PCA') # Perform PCA on the homework data
    da.pca_process(compulsory_activities_data, 0.95, 'Compulsory_activities_PCA') # Perform PCA on the compulsory activities data
    da.pca_process(optional_activities_data, 0.95, 'Optional_activities_PCA') # Perform PCA on the optional activities data

    # Find if the students that had a good homework mean passed the exams (Two sample t-test)
    # (H0: The students that had a good homework mean passed the exams, H1: The students that had a good homework mean did not pass the exams)
    mean_homework_mark, exam_result = dataProcessing.create_hypothesis_testing_dataset()
    print("\nMean homework mark:\n", mean_homework_mark.head())
    print("\nExam result:\n", exam_result.head())

    print("Hypothesis testing results:\n", da.hypothesis_testing(mean_homework_mark, exam_result)) # Perform a t-test on the two groups

    # Try to predict the exams mark based on their performance in homework and activities using Gradient Descent:
    X, X_scaled, y = da.create_gradient_descent_dataset(original_data) # Create the dataset for gradient descent
    predictions = da.gradient_descent_process(X_scaled, y) # Perform gradient descent on the dataset and keep predictions

    # Create the Pass Actual and Pass Predicted columns where if Exam results is >= 5 then the value is 0 else is 1
    predictions_df = pd.DataFrame({'predictions': predictions})
    original_data['Pass Actual'] = original_data.apply(lambda row: 0 if row['Exam'] >= 5 else 1, axis = 1)
    original_data['Pass Predicted'] = predictions_df['predictions'].apply(lambda x: 0 if x >= 5 else 1)

    da.confusion_matrix_application(original_data['Pass Actual'], original_data['Pass Predicted']) # Create the confusion matrix of predicted and actual pass results:

    # Make the same with the previous process, but using a "Handmade" Gradient Descent function:
    X = np.c_[np.ones(X.shape[0]), X] # Add a column of ones for the bias term

    # Initialize parameters
    theta = np.zeros(X.shape[1])
    alpha = 0.01 # The learning rate
    num_iters = 1000 # The number of iterations

    theta, cost_history = da.custom_gradient_descent(X, y, theta, alpha, num_iters)
    print("Theta (model parameters): ", theta)
    print("Final cost: ", cost_history[-1])

if __name__ == '__main__':
    main()
