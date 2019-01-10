from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

import mod_scorers

paramGrid = [
    { 
        "normalize": [
            True, False 
        ]
    }
]

def trainModel( features, arousalLabels, valenceLabels ):
    gridSearchArousalLr = GridSearchCV( LinearRegression(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchValenceLr = GridSearchCV( LinearRegression(), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchArousalLr.fit( features, arousalLabels )
    gridSearchValenceLr.fit( features, valenceLabels )

    return gridSearchArousalLr, gridSearchValenceLr