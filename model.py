import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from utils import get_data

data_dir = './traindata'

data, labels, class_names = get_data(data_dir)
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

classifier = SVC()

parameters = [{
    'gamma': [0.01, 0.001, 0.001],
    'C': [1, 10, 100, 1000]
}]

grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(train_x, train_y)
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(test_x)
acc_score = accuracy_score(y_prediction, test_y)
print(f"Accuracy score: {acc_score}%")
if acc_score > 0.98:
    pickle.dump(best_estimator, open('./output_model/model.p', 'wb'))
