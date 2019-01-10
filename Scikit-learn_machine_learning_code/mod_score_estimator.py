import numpy as np

import mod_logger as logger
import mod_smoothing as smoothing

def scoreEstimator( estimator, scorer, features, labels, enableSmoothing=True ):
    predictions = estimator.predict( features )

    if enableSmoothing == True:
        predictions = smoothing.movingAverageFilter( predictions, 3 )
        labels = smoothing.movingAverageFilter( labels, 3 )

    estimatorScore = scorer( predictions, labels )
    return estimatorScore