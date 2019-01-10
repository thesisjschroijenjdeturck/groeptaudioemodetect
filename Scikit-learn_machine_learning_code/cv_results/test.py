import numpy as np
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt

def printColsOverCol( df, independentCol, independentColLabel, valueLabel, dependentCols, dependentColLabels ):
    independentCol = df[ independentCol ].values.astype( float )
    fig = plt.figure()
    sp = fig.add_subplot( 111 )
    sp.set_title( 'Ordinary linear regression scores in function of alpha' )
    for dependentCol, dependentColLabel in zip( dependentCols, dependentColLabels ):
        dependentCol = df[ dependentCol ].values
        sp.plot( np.log10( independentCol ), dependentCol, label=dependentColLabel )
    plt.xlabel( independentColLabel )
    plt.ylabel( valueLabel )
    plt.legend()
    plt.show()

ridgeRegArousalDf = pd.DataFrame( joblib.load( 'sewa_dataset_lasso_reg_arousal_cv_results.pickle' ) )
printColsOverCol( ridgeRegArousalDf, 'param_alpha', 'Log base 10(alpha)', 'CCC score', [ 'mean_train_score', 'mean_test_score' ], [ 'Mean train score', 'Mean test score' ] )