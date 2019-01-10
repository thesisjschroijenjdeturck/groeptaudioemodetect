import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.externals import joblib

import mod_logger as logger

MOD_PCA_TAG = "MOD_PCA"

def buildPcaPipeline( explainedVarianceFraction, estimatorName, estimator ):
    return Pipeline( steps=[ ( "pca", PCA( n_components=explainedVarianceFraction, svd_solver="full" ) ), ( estimatorName, estimator ) ] )
    
def transformToPrincipalComponents( explainedVarianceFraction, featuresTrain, featuresTest = None, log = True, dumpTransformer = True, dumpedTransformerFilename = "principal_component_transformer.pickle" ):
    pca = PCA( n_components=explainedVarianceFraction, svd_solver="full" )
    pca = pca.fit( featuresTrain )
    
    if log == True:
        logger.logMessage( MOD_PCA_TAG, "# of principal components found was {}".format( pca.n_components_ ) )
        logger.logMessage( MOD_PCA_TAG, "Explained variance was {}".format( np.sum( pca.explained_variance_ratio_ ) ) )

    if dumpTransformer == True:
        joblib.dump( pca, "saved_principal_component_transformers/" + dumpedTransformerFilename, protocol = 2 )

    if featuresTest is None:
        return pca.transform( featuresTrain )
    else:
        return pca.transform( featuresTrain ), pca.transform( featuresTest )