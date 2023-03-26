import pickle
import pandas as pd

load_model = pickle.load(open('model.pkl', 'rb'))
data_test = pd.read_csv('test/data_test.csv')