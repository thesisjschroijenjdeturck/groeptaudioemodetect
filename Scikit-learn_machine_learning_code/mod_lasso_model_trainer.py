from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

import mod_scorers

paramGrid = [
    { 
        "normalize": [
            True, False 
        ],
        "alpha": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10,
            100,
            1000,
            10000
        ]
    }
]

def trainModel( features, arousalLabels, valenceLabels ):
    gridSearchArousalLassoReg = GridSearchCV( Lasso(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchValenceLassoReg = GridSearchCV( Lasso(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchArousalLassoReg.fit( features, arousalLabels )
    gridSearchValenceLassoReg.fit( features, valenceLabels )

    return gridSearchArousalLassoReg, gridSearchValenceLassoReg