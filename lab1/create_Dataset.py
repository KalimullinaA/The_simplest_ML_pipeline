import gdown as gdown
import pandas as pd
from sklearn.model_selection import train_test_split
import os


gdown.download(id="12Do3EWfXLHY0dmKq6YckeF0LOa2M47gD", output="./Data/Salary Data.csv", quiet=False)
dataframe = pd.read_csv('Data/Salary Data.csv', delimiter = ',', index_col = 0)
# dataframe.dropna(inplace=True)
# train, test = train_test_split(dataframe, test_size=0.2)

# train['Gender'] = train['Gender'].apply(lambda x:0 if 'male' else 1)
# test['Gender'] = test['Gender'].apply(lambda x:0 if 'male' else 1)
#
# train[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']].to_csv('data_train.csv', index = False)
# os.replace('data_train.csv','train/data_train.csv')
# test[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']].to_csv('data_test.csv', index = False)
# os.replace('data_test.csv','test/data_test.csv')

#делим данные на тренировочные и тестовые
X_train, X_test, Y_train, Y_test = train_test_split(
    dataframe[['Gender', 'Education Level', 'Job Title', 'Years of Experience','Salary']],
    dataframe[['Years of Experience']],
    test_size = 0.20,
    random_state = 42
)
#сохраняем файлы в папках train и test
X_train.to_csv('train/X_train.csv', index=True)
X_test.to_csv('test/X_test.csv', index=True)
Y_train.to_csv('train/Y_train.csv', index=False)
Y_test.to_csv('test/Y_test.csv', index=False)