from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import os


data_train = pd.read_csv('train/data_train.csv')
data_test = pd.read_csv('test/data_test.csv')

DF_full = pd.concat([data_train, data_test]) # объединить DF_train и DF_test в один ДатаФрейм


DF_full.drop('Job Title', axis=1, inplace=True)

cat_columns = []
num_columns = []
for column_name in DF_full.columns:
    if (DF_full[column_name].dtypes == object): # проверяем тип данных для каждой колонки
        cat_columns +=[column_name] # если тип объект - то складываем в категориальные данные
    else:
        num_columns +=[column_name] # иначе - числовые
print('Категориальные данные:\t ', cat_columns, '\n Число столблцов = ',len(cat_columns))
print('Числовые данные:\t ',  num_columns, '\n Число столблцов = ',len(num_columns))

DF_ohe = pd.get_dummies(DF_full[cat_columns]) # One-hot кодирование многозначных признаков
DF_full = DF_full.join(DF_ohe)
DF_full.drop(columns=['Education Level'], inplace=True)
DF_full['Salary'] = DF_full['Salary'].fillna(DF_full.Salary.mean())


train = DF_full.iloc[0:data_train.shape[0],:] # Разбиваем данные на Тренировочную и Тестовую 0-300,:
test = DF_full.iloc[data_train.shape[0]:,:]#300-end


X_train = train.drop([i for i in cat_columns if i in ['Job Title']], axis=1)
y_train = data_train['Salary'].values

X_test = test.drop([i for i in cat_columns if i in ['Job Title']], axis=1)


X_test[X_test.columns.tolist()].to_csv('data_test.csv', index = False)
os.replace('data_test.csv','test/data_test.csv')
model = LogisticRegression(max_iter=100).fit(X_train, y_train)


pickle.dump(model, open('model.pkl', 'wb'))