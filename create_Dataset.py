import pandas as pd
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv(r"C:\Users\Admin\Documents\GitHub\MLOps\data\Salary Data.csv")
dataframe.dropna(inplace=True)
train, test = train_test_split(dataframe, test_size=0.2)

train['Gender'] = train['Gender'].apply(lambda x:0 if 'male' else 1)
test['Gender'] = test['Gender'].apply(lambda x:0 if 'male' else 1)

train[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']].to_csv('data_train.csv', index = False)
test[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']].to_csv('data_test.csv', index = False)