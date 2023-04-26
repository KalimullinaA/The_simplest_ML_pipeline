import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import os

# Загружаем модель машинного обучения из файла
filename = 'model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Проверяем точность модели на данных из папки "test"
test_data = []
for file in os.listdir('test'):
    if file.endswith('.csv'):
        data = pd.read_csv(f'test/{file}')
        X = data[["Gender","Years of Experience","Salary","Education Level_Bachelor's","Education Level_Master's","Education Level_PhD"]]
        y = data['Salary']
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        test_data.append((file, mse))
print('Test data:')
for data in test_data:
    print(f'{data[0]} - MSE: {data[1]}')