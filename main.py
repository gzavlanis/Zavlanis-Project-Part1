from Data_Processing import DataProcessing
import Data_Analytics as da

def main():
    dataProcessing = DataProcessing()
    exams_data, homework_data, compulsory_activities_data, optional_activities_data = dataProcessing.get_formatted_data()
    print("Exams data:\n", exams_data.head(5))
    print("\nHomework data:\n", homework_data.head(5))
    print("\nCompulsory activities data:\n", compulsory_activities_data.head(5))
    print("\nOptional activities data:\n", optional_activities_data.head(5))

    exams_mean_values = da.find_mean_values(exams_data)
    homework_mean_values = da.find_mean_values(homework_data)
    compulsory_activities_mean_values = da.find_mean_values(compulsory_activities_data)
    optional_activities_mean_values = da.find_mean_values(optional_activities_data)

    exams_standard_deviation = da.find_standard_deviation(exams_data)
    homework_standard_deviation = da.find_standard_deviation(homework_data)
    compulsory_activities_standard_deviation = da.find_standard_deviation(compulsory_activities_data)
    optional_activities_standard_deviation = da.find_standard_deviation(optional_activities_data)

    print("\nExams mean values:\n", exams_mean_values)
    print("\nHomework mean values:\n", homework_mean_values)
    print("\nCompulsory activities mean values:\n", compulsory_activities_mean_values)
    print("\nOptional activities mean values:\n", optional_activities_mean_values)

    print("\nExams standard deviation:\n", exams_standard_deviation)
    print("\nHomework standard deviation:\n", homework_standard_deviation)
    print("\nCompulsory activities standard deviation:\n", compulsory_activities_standard_deviation)
    print("\nOptional activities standard deviation:\n", optional_activities_standard_deviation)

    da.pca_process(exams_data, 0.95) # Perform PCA on the exams data
    da.pca_process(homework_data, 0.95) # Perform PCA on the homework data
    da.pca_process(compulsory_activities_data, 0.95) # Perform PCA on the compulsory activities data
    da.pca_process(optional_activities_data, 0.95) # Perform PCA on the optional activities data

if __name__ == '__main__':
    main()
