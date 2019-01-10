import random
import time

import mysql.connector
import numpy

import mod_logger

module_tag = "MOD_DATA_LOADER"

databaseConfig = {
  'user': 'root',
  'password': 'eax;ebx;824',
  'host': 'localhost',
  'port': '3306',
  'database': 'thesis'
}

featuresFetchQueryTemplate = \
  "SELECT " \
  "MfccCoefficient0, MfccCoefficient1, MfccCoefficient2, MfccCoefficient3, MfccCoefficient4, MfccCoefficient5, MfccCoefficient6, MfccCoefficient7, MfccCoefficient8, MfccCoefficient9, MfccCoefficient10, MfccCoefficient11, MfccCoefficient12, " \
  "MfccDeltaCoefficient0, MfccDeltaCoefficient1, MfccDeltaCoefficient2, MfccDeltaCoefficient3, MfccDeltaCoefficient4, MfccDeltaCoefficient5, MfccDeltaCoefficient6, MfccDeltaCoefficient7, MfccDeltaCoefficient8, MfccDeltaCoefficient9, MfccDeltaCoefficient10, MfccDeltaCoefficient11, MfccDeltaCoefficient12, " \
  "MfccDeltaDeltaCoefficient0, MfccDeltaDeltaCoefficient1, MfccDeltaDeltaCoefficient2, MfccDeltaDeltaCoefficient3, MfccDeltaDeltaCoefficient4, MfccDeltaDeltaCoefficient5, MfccDeltaDeltaCoefficient6, MfccDeltaDeltaCoefficient7, MfccDeltaDeltaCoefficient8, MfccDeltaDeltaCoefficient9, MfccDeltaDeltaCoefficient10, MfccDeltaDeltaCoefficient11, MfccDeltaDeltaCoefficient12 " \
  "FROM sewadataset WHERE LabeledFeatureVectorID = %d AND FeatureVectorSet = '%s'"
labelsFetchQueryTemplate = \
  "SELECT Arousal, Valence FROM sewadataset WHERE LabeledFeatureVectorID = %d AND FeatureVectorSet = '%s'"

numberOfFeatures = 39

def loadData( setToLoad, setStartId, setSize, percentageToLoad, randomSeed = 42 ):
    random.seed( randomSeed ) # To make program results repeatable

    dbConnection = mysql.connector.connect( **databaseConfig )

    subsetSize = int( round( percentageToLoad * setSize ) )
    randomLabeledFeatureVectorIds = random.sample( range( setStartId, ( setStartId + setSize ) ), subsetSize )    
    features = numpy.zeros( ( subsetSize, numberOfFeatures ) )
    arousalLabels = numpy.zeros( subsetSize )
    valenceLabels = numpy.zeros( subsetSize )

    mod_logger.logMessage( module_tag, "Loading features and labels of set {} from database ...".format( setToLoad) )
    t1 = time.time()
    dbCursor = dbConnection.cursor()
    currentIndex = 0
    for id in randomLabeledFeatureVectorIds:
        dbCursor.execute( ( featuresFetchQueryTemplate ) % ( id, setToLoad ) )
        currentFeatureVector = dbCursor.fetchone()
        features[ currentIndex ] = numpy.array( currentFeatureVector )
    
        dbCursor.execute( ( labelsFetchQueryTemplate ) % ( id, setToLoad ) )
        currentLabels = dbCursor.fetchone()
        arousalLabels[ currentIndex ] = currentLabels[ 0 ]
        valenceLabels[ currentIndex ] = currentLabels[ 1 ]

        currentIndex = currentIndex + 1

    t2 = time.time()
    t = t2 - t1
    mod_logger.logMessage( module_tag, "Features and labels of set %s loaded from database in %d seconds" % ( setToLoad, t ) )

    mod_logger.logMessage( module_tag, "Features have shape {}".format( features.shape ) )
    mod_logger.logMessage( module_tag, "Arousal labels have shape {}".format( arousalLabels.shape ) )
    mod_logger.logMessage( module_tag, "Valence labels have shape {}".format( valenceLabels.shape ) )

    return features, arousalLabels, valenceLabels