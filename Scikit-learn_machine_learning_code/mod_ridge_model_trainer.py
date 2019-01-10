from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

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
    gridSearchArousalRidgeReg = GridSearchCV( Ridge(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchValenceRidgeReg = GridSearchCV( Ridge(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchArousalRidgeReg.fit( features, arousalLabels )
    gridSearchValenceRidgeReg.fit( features, valenceLabels )

    return gridSearchArousalRidgeReg, gridSearchValenceRidgeReg