import pandas as pd

class DataProcessing:
    def __init__(self):
        exam_columns = 'A:B' # Define the columns of the data
        homework_columns = 'C:F'
        compulsory_activities_columns = 'G:N'
        optional_activities_columns = 'O:X'

        self.exams_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = exam_columns) # Load the data from the excel file
        self.homework_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = homework_columns, names = ['1', '2', '3', '4']) # Load the data from the excel file
        self.compulsory_activities_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = compulsory_activities_columns, names = ['1', '2', '3', '4', '5', '6', '7', '8']) # Load the data from the excel file
        self.optional_activities_data = pd.read_excel('./Data/grades.xlsx', skiprows = 1, usecols = optional_activities_columns, names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) # Load the data from the excel file

    def convert_to_numeric(self):
        self.exams_data.replace({'-': 0, -1: 0}, inplace = True) # Replace the missing values with 0
        self.homework_data.replace({'-': 0, -1: 0}, inplace = True)
        self.compulsory_activities_data.replace({'-': 0, -1: 0}, inplace = True)
        self.optional_activities_data.replace({'-': 0, -1: 0}, inplace = True)

    def get_formatted_data(self):
        self.convert_to_numeric()
        return self.exams_data, self.homework_data, self.compulsory_activities_data, self.optional_activities_data
