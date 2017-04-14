import sys
import json
import numpy as np
import random as rnd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.metrics import precision_recall_fscore_support


class TrainDataConstructor():
    def __init__(self, file):
        self.scaler = StandardScaler()

        with open(file, 'rt') as trainfile:
            data = trainfile.readlines()

        for ind in range(len(data)):
            data[ind] = json.loads(data[ind])

        self.features = []
        self.targets = []
        for node in data:
            node_features = [0] * len(node['stat'].keys())
            for tr_type in node['stat']:
                node_features[int(tr_type) - 1] = node['stat'][tr_type]['val']
            self.features.append(node_features)
            self.targets.append(round(float(node['age'])))

        self.scaler.fit(np.array(self.features))

    def construct(self, transform=False, without_non_ct=True):
        if without_non_ct:
            self.features, self.targets = zip(*filter(lambda x: x[1] >= 0, zip(self.features, self.targets)))

        if transform:
            self.features = self.scaler.transform(np.array(self.features))

        return np.array(self.features), np.array(self.targets)


class Estimator():
    def __init__(self):
        self.sgd = SGDClassifier(penalty='elasticnet', l1_ratio=0.15, alpha=1.0e-4)
        self.rbf_svc = SVC(decision_function_shape='ovr')
        self.lin_svc = LinearSVC()

    def train(self, features, targets):
        self.sgd.fit(features, targets)
        self.rbf_svc.fit(features, targets)
        self.lin_svc.fit(features, targets)

    def test(self, features, targets):
        res = cross_val_score(self.sgd, features, targets, cv=5, scoring='f1')
        print("SGDClassifier:\n\t{0}".format(res))
        res = cross_val_score(self.rbf_svc, features, targets, cv=5, scoring='f1')
        print("SVC with RBF cernel:\n\t{0}".format(res))
        res = cross_val_score(self.lin_svc, features, targets, cv=5, scoring='f1')
        print("SVC with linear kernel:\n\t{0}".format(res))


class GridSearch():
    def __init__(self):
        estimator = SGDClassifier(penalty='elasticnet')
        est_parameters = {'l1_ratio': [0, 0.15, 0.3, 0.5, 0.7, 1],
                          'alpha': list(10.0 ** -np.arange(1, 7))}

        self.clf = GridSearchCV(estimator, est_parameters, n_jobs=-1, pre_dispatch=2)
        self.res = []

    def run(self, features, targets):
        self.clf.fit(features, targets)
        self.res = self.clf.best_params_

    def test(self, features, targets):
        y_true, y_pred = targets, self.clf.predict(features)
        res = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print("{0}".format(res))


def main(trainfile):
    tdc = TrainDataConstructor(trainfile)

    est = Estimator()

    # without transform
    features, targets = tdc.construct(transform=False)
    print("Without transforming:")
    est.test(features, targets)

    # with transform
    features, targets = tdc.construct(transform=True)
    print("With transforming:")
    est.test(features, targets)


if __name__ == "__main__":
    trainfile = sys.argv[1]
    main(trainfile)
