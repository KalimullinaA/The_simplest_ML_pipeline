import pickle
import pandas as pd


load_model = pickle.load(open('model.pkl', 'rb'))
data_test = pd.read_csv('train/y_train.csv')
X_test = data_test[data_test.columns.tolist()].values
print(load_model.predict(X_test[0:1]))
