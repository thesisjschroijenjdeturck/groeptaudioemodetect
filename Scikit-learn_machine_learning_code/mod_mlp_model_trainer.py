from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

import mod_pca
import mod_scorers

def testCallback( avg_valid_error, avg_train_error, **_ ):
    print( "Avg. valid. error = {0}, avg. train error = {1}" % ( avg_valid_error, avg_train_error ) )

paramGrid = [
    {
        "mlp__hidden_layer_sizes": [
            ( 39 ),
            ( 40 ),
            ( 42 ),
            ( 44 ),
            ( 46 ),
            ( 48 ),
            ( 50 ),
            ( 64 ),
            ( 128 ),
            ( 39, 39 ),
            ( 40, 40 ),
            ( 42, 42 ),
            ( 44, 44 ),
            ( 46, 46 ),
            ( 48, 48 ),
            ( 50, 50 ),
            ( 64, 64 ),
            ( 128, 128 ),
        ],
        "mlp__activation": [ "relu" ],
        "mlp__solver": [ "adam" ],
        "mlp__alpha": [ 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001 ],
        "mlp__learning_rate_init": [ 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001 ],
        "mlp__learning_rate": [ "constant", "invscaling" ],
        "mlp__callbacks": [{ 'on_epoch_finish': testCallback }]
    }
]

def trainModel( features, arousalLabels, valenceLabels ):
    gridSearchArousalMlp = GridSearchCV( mod_pca.buildPcaPipeline( "mlp", MLPRegressor() ), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchValenceMlp = GridSearchCV( mod_pca.buildPcaPipeline( "mlp", MLPRegressor() ), paramGrid, cv=10, scoring=mod_scorers.CccScore, n_jobs=-1 )
    gridSearchArousalMlp.fit( features, arousalLabels )
    gridSearchValenceMlp.fit( features, valenceLabels )
    
    return gridSearchArousalMlp, gridSearchValenceMlp