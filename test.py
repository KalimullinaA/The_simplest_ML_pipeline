import pandas as pd
import pickle
from sklearn.metrics import f1_score


LogReg = pickle.load(open('model.pkl', 'rb'))
X_test = pd.read_csv('test/X_test.csv', delimiter = ',')
Y_test = pd.read_csv('test/Y_test.csv', delimiter = ',')
y_preds = LogReg.predict(X_test)
print(f'f1_score: {f1_score(Y_test, y_preds, average="micro")}')