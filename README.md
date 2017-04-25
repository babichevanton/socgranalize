# socgranalize
Classifying social graph nodes with help of graph structure features.

Grid search for age classification task.

Estimators:
1) SGDClassifier
	-loss
	-penalty - ElasticNet, hyper r1 & r2
	-alpha


Current efficiency at age determination(5-fold CV, f1-score):

Without transforming:
SGDClassifier:
	0.66666667  0.11584875  0.66488095  0.66686408  0.23695346
SVC with RBF cernel:
	0.52185549  0.53452116  0.50705508  0.54283217  0.55154415
SVC with linear kernel:
	0.42285104  0.27576792  0.66627149  0.6498899   0.48869223

With transforming (StandardScaler):
SGDClassifier:
	0.25570776  0.05016722  0.05245347  0.64120603  0.66106965
SVC with RBF cernel:
	0.62920569  0.63934981  0.62543433  0.62478062  0.63496408
SVC with linear kernel:
	0.6126636   0.61732725  0.61039886  0.58836944  0.60956618


current efficiency at acount status determination(1-fold):

First training samples (~250 '1' vs ~25000 '0'):
With transforming:
SGDClassifier:
    prec: 0.2159    rec: 0.3363    f1: 0.2630
GradientBoostClassifier:
    prec: 0.3889    rec: 0.0619    f1: 0.1069
RandomForestClassifier with unbalanced training samples:
	prec: 0.3953    rec: 0.1504    f1: 0.2179
RandomForestClassifier with balanced training samples:
	prec: 0.1294    rec: 0.4513    f1: 0.2012

Without transforming: complete
SGDClassifier:
	prec: 1.0000    rec: 0.0085    f1: 0.0168
GradientBoostClassifier:
	prec: 0.4211    rec: 0.0678    f1: 0.1168
RandomForestClassifier with unbalanced training samples:
	prec: 0.3438    rec: 0.0932    f1: 0.1467
RandomForestClassifier with balanced training samples:
	prec: 0.1161    rec: 0.4576    f1: 0.1852



