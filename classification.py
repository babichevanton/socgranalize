import sys
import json
from itertools import cycle

import numpy as np
from scipy import interp

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc


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
            self.targets.append(int(node['label']))

        self.scaler.fit(np.array(self.features))

    def construct(self, transform=False):
        if transform:
            self.features = self.scaler.transform(np.array(self.features))

        return np.array(self.features), np.array(self.targets)


class Estimator():
    def __init__(self):
        self.sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', l1_ratio=0.15, alpha=1.0e-4)
        self.rbf_svc = SVC(decision_function_shape='ovr')
        self.unbalanced_forest = RandomForestClassifier(class_weight=None)
        self.balanced_forest = RandomForestClassifier(class_weight='balanced')

    def _onefold_roc(self, ind, clf, X_train, X_test, y_train, y_test, mean_tpr, mean_fpr):
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr[ind] += interp(mean_fpr, fpr, tpr)
        mean_tpr[ind][0] = 0.0
        return mean_tpr

    def test(self, features, targets, resfile):
        cv = StratifiedKFold(n_splits=6)

        mean_tpr = [0.0] * 4
        mean_fpr = np.linspace(0, 1, 100)

        colors = cycle(['cyan', 'seagreen', 'blue', 'darkorange'])
        lw = 2

        for train, test in cv.split(features, targets):
            mean_tpr = self._onefold_roc(0,
                                         self.sgd,
                                         features[train],
                                         features[test],
                                         targets[train],
                                         targets[test],
                                         mean_tpr,
                                         mean_fpr)
            mean_tpr = self._onefold_roc(1,
                                         self.rbf_svc,
                                         features[train],
                                         features[test],
                                         targets[train],
                                         targets[test],
                                         mean_tpr,
                                         mean_fpr)
            mean_tpr = self._onefold_roc(2,
                                         self.unbalanced_forest,
                                         features[train],
                                         features[test],
                                         targets[train],
                                         targets[test],
                                         mean_tpr,
                                         mean_fpr)
            mean_tpr = self._onefold_roc(3,
                                         self.balanced_forest,
                                         features[train],
                                         features[test],
                                         targets[train],
                                         targets[test],
                                         mean_tpr,
                                         mean_fpr)

        for ind, color, clf_name in zip([0,1,2,3], colors, ['SGD', 'SVM', 'RF_ub','RF_b']):
            mean_tpr[ind] /= cv.get_n_splits(features, targets)
            mean_tpr[ind][-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr[ind])
            plt.plot(mean_fpr, mean_tpr[ind], color=color,
                     label='{0} ROC (area = {1})'.format(clf_name, mean_auc), lw=lw)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(resfile)


def main(trainfile, resfile):
    tdc = TrainDataConstructor(trainfile)

    est = Estimator()

    for transform in [False, True]:
        if transform:
            features, targets = tdc.construct(transform=True)
            est.test(features, targets, resfile)
            print("Without transforming: complete")
        else:
            features, targets = tdc.construct(transform=False)
            est.test(features, targets, resfile)
            print("With transforming: complete")


if __name__ == "__main__":
    trainfile = sys.argv[1]
    resfile = sys.argv[2]
    main(trainfile, resfile)
