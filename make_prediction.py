import pickle
from train__model import X_test


load_model = pickle.load(open('model.pkl', 'rb'))
print(load_model.predict(X_test[0:1]))

