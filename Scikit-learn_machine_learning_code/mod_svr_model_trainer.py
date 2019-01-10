from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import mod_scorers

paramGrid = [
    { "kernel": [ 'linear' ], "degree": range( 1, 11 ), "C": [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ], "max_iter": [ 10000, 100000 ] },
    { "kernel": [ 'poly', 'rbf', 'sigmoid' ], "gamma": [ 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001 ], "C": [ 0.001, 0.01, 0.1, 1, 10, 100, 1000 ], "max_iter": [ 10000, 100000 ] }
]

def trainModel( features, arousalLabels, valenceLabels ):
    gridSearchArousalSvr = GridSearchCV( SVR(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1, verbose=10 )
    gridSearchValenceSvr = GridSearchCV( SVR(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1, verbose=10 )
    gridSearchArousalSvr.fit( features, arousalLabels )
    gridSearchValenceSvr.fit( features, valenceLabels )
    
    return gridSearchArousalSvr, gridSearchValenceSvr