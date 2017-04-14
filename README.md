# socgranalize
Classifying social graph nodes with help of graph structure features.

Grid search for age classification task.

Estimators:
1) SGDClassifier
    -loss
    -penalty - ElasticNet, hyper r1 & r2
    -alpha


Current efficiency (5-fold CV, f1-score):

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


