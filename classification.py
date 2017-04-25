import sys
import json
from itertools import cycle

import numpy as np
from scipy import interp

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



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
        self.rbf_svc = SVC(probability=True)
        self.unbalanced_forest = RandomForestClassifier(class_weight=None)
        self.balanced_forest = RandomForestClassifier(class_weight='balanced')
        self.grad_boost = GradientBoostingClassifier()

    def _onefold_test(self, ind, clf, X_train, X_test, y_train, y_test, mean_tpr, mean_fpr):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print('prec: {0}\nrec: {1}\nf1: {2}'.format(precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)))
        probas_ = clf.predict_proba(X_test)
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr[ind] += interp(mean_fpr, fpr, tpr)
        mean_tpr[ind][0] = 0.0
        return mean_tpr

    def test(self, features, targets, resfile):
        mean_tpr = [0.0] * 4
        mean_fpr = np.linspace(0, 1, 100)

        colors = cycle(['cyan', 'seagreen', 'blue', 'darkorange', 'red'])
        lw = 2

        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

        clfs = []
        mean_tpr = self._onefold_test(0,
                                      self.sgd,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      mean_tpr,
                                      mean_fpr)
        clfs.append(0)
        print('SGD complete')
        mean_tpr = self._onefold_test(1,
                                      self.grad_boost,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      mean_tpr,
                                      mean_fpr)
        clfs.append(1)
        print('GB complete')
        mean_tpr = self._onefold_test(2,
                                      self.unbalanced_forest,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      mean_tpr,
                                      mean_fpr)
        clfs.append(2)
        print('unb RF complete')
        mean_tpr = self._onefold_test(3,
                                      self.balanced_forest,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      mean_tpr,
                                      mean_fpr)
        clfs.append(3)
        print('b RF complete')

        for ind, color, clf_name in zip(clfs, colors, ['SGD', 'GB', 'RF_ub','RF_b']):
            mean_tpr[ind][-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr[ind])
            plt.plot(mean_fpr, mean_tpr[ind], color=color,
                     label='%s ROC (area = %.4f)' % (clf_name, mean_auc), lw=lw)

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

    name1, name2 = resfile.split('.')

    for transform in [False, True]:
        if transform:
            features, targets = tdc.construct(transform=True)
            plt.figure(1)
            est.test(features, targets, '{0}_tr.{1}'.format(name1, name2))
            print("Without transforming: complete")
        else:
            features, targets = tdc.construct(transform=False)
            plt.figure(2)
            est.test(features, targets, '{0}_ntr.{1}'.format(name1, name2))
            print("With transforming: complete")


if __name__ == "__main__":
    trainfile = sys.argv[1]
    resfile = sys.argv[2]
    main(trainfile, resfile)
