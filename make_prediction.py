import pickle

import pandas as pd

# from train__model import X_test


load_model = pickle.load(open('model.pkl', 'rb'))
data_test = pd.read_csv('test/data_test.csv')
X_test = data_test[data_test.columns.tolist()].values
print(load_model.predict(X_test[0:1]))

