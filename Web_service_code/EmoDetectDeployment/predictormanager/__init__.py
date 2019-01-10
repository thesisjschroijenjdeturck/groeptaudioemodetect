import random
import math
import os

import numpy as np
from flask import Flask, request, Response, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from sklearn.externals import joblib
from typing import List, Dict

app = Flask( __name__ )
api = Api( app )

app.config[ 'SQLALCHEMY_DATABASE_URI' ] = 'mysql+pymysql://emodetect:GroepTEmoDetect@localhost:3306/'
app.config[ 'SECRET_KEY' ] = '8bb96e26e4cdbe60cb41c2e8706f3270'

db = SQLAlchemy( app )

from predictormanager import models

def LOGGER(msg):
    with open( '/home/deploy/EmoDetectDeployment/LOG.TXT', 'a' ) as f:
        f.write( msg )

class PredictorAPI( Resource ):
    def __dictionaryToNumpyArray( self, theDictionary ):
        return( np.array( list( theDictionary.values() ) ) )

    def __storeInformation( self, recordingID, requestFeatureVector, arousalPrediction, valencePrediction ):
        newMfccFeature = models.MfccFeature()
        newMfccFeature.RecordingID = recordingID
        newMfccFeature.FrameOffset = requestFeatureVector[ "frameOffset" ]
        newMfccFeature.MfccCoefficient0 = requestFeatureVector[ "features" ][ 0 ]
        newMfccFeature.MfccCoefficient1 = requestFeatureVector[ "features" ][ 1 ]
        newMfccFeature.MfccCoefficient2 = requestFeatureVector[ "features" ][ 2 ]
        newMfccFeature.MfccCoefficient3 = requestFeatureVector[ "features" ][ 3 ]
        newMfccFeature.MfccCoefficient4 = requestFeatureVector[ "features" ][ 4 ]
        newMfccFeature.MfccCoefficient5 = requestFeatureVector[ "features" ][ 5 ]
        newMfccFeature.MfccCoefficient6 = requestFeatureVector[ "features" ][ 6 ]
        newMfccFeature.MfccCoefficient7 = requestFeatureVector[ "features" ][ 7 ]
        newMfccFeature.MfccCoefficient8 = requestFeatureVector[ "features" ][ 8 ]
        newMfccFeature.MfccCoefficient9 = requestFeatureVector[ "features" ][ 9 ]
        newMfccFeature.MfccCoefficient10 = requestFeatureVector[ "features" ][ 10 ]
        newMfccFeature.MfccCoefficient11 = requestFeatureVector[ "features" ][ 11 ]
        newMfccFeature.MfccCoefficient12 = requestFeatureVector[ "features" ][ 12 ]
        newMfccFeature.MfccDeltaCoefficient0 = requestFeatureVector[ "features" ][ 13 ]
        newMfccFeature.MfccDeltaCoefficient1 = requestFeatureVector[ "features" ][ 14 ]
        newMfccFeature.MfccDeltaCoefficient2 = requestFeatureVector[ "features" ][ 15 ]
        newMfccFeature.MfccDeltaCoefficient3 = requestFeatureVector[ "features" ][ 16 ]
        newMfccFeature.MfccDeltaCoefficient4 = requestFeatureVector[ "features" ][ 17 ]
        newMfccFeature.MfccDeltaCoefficient5 = requestFeatureVector[ "features" ][ 18 ]
        newMfccFeature.MfccDeltaCoefficient6 = requestFeatureVector[ "features" ][ 19 ]
        newMfccFeature.MfccDeltaCoefficient7 = requestFeatureVector[ "features" ][ 20 ]
        newMfccFeature.MfccDeltaCoefficient8 = requestFeatureVector[ "features" ][ 21 ]
        newMfccFeature.MfccDeltaCoefficient9 = requestFeatureVector[ "features" ][ 22 ]
        newMfccFeature.MfccDeltaCoefficient10 = requestFeatureVector[ "features" ][ 23 ]
        newMfccFeature.MfccDeltaCoefficient11 = requestFeatureVector[ "features" ][ 24 ]
        newMfccFeature.MfccDeltaCoefficient12 = requestFeatureVector[ "features" ][ 25 ]
        newMfccFeature.MfccDeltaDeltaCoefficient0 = requestFeatureVector[ "features" ][ 26 ]
        newMfccFeature.MfccDeltaDeltaCoefficient1 = requestFeatureVector[ "features" ][ 27 ]
        newMfccFeature.MfccDeltaDeltaCoefficient2 = requestFeatureVector[ "features" ][ 28 ]
        newMfccFeature.MfccDeltaDeltaCoefficient3 = requestFeatureVector[ "features" ][ 29 ]
        newMfccFeature.MfccDeltaDeltaCoefficient4 = requestFeatureVector[ "features" ][ 30 ]
        newMfccFeature.MfccDeltaDeltaCoefficient5 = requestFeatureVector[ "features" ][ 31 ]
        newMfccFeature.MfccDeltaDeltaCoefficient6 = requestFeatureVector[ "features" ][ 32 ]
        newMfccFeature.MfccDeltaDeltaCoefficient7 = requestFeatureVector[ "features" ][ 33 ]
        newMfccFeature.MfccDeltaDeltaCoefficient8 = requestFeatureVector[ "features" ][ 34 ]
        newMfccFeature.MfccDeltaDeltaCoefficient9 = requestFeatureVector[ "features" ][ 35 ]
        newMfccFeature.MfccDeltaDeltaCoefficient10 = requestFeatureVector[ "features" ][ 36 ]
        newMfccFeature.MfccDeltaDeltaCoefficient11 = requestFeatureVector[ "features" ][ 37 ]
        newMfccFeature.MfccDeltaDeltaCoefficient12 = requestFeatureVector[ "features" ][ 38 ]
        newMfccFeature.ArousalPrediction = arousalPrediction
        newMfccFeature.ValencePrediction = valencePrediction
        db.session.add( newMfccFeature )
        db.session.commit()

    def __loadPredictor( self ):
         selectedPredictor = models.Predictor.query.filter_by( active = True ).first()

         try:
             arousalPredictor = joblib.load( os.path.join( app.root_path, "uploaded_predictors", "arousal", selectedPredictor.arousalPredictorFilename ) )
         except:
             arousalPredictor = None
         try:
             valencePredictor = joblib.load( os.path.join( app.root_path, "uploaded_predictors", "valence", selectedPredictor.valencePredictorFilename ) )
         except:
             valencePredictor = None
         try:
             principalComponentTransformer = joblib.load( os.path.join( app.root_path, "uploaded_predictors", "principal_component_transformers", selectedPredictor.principalComponentTransformerFilename ) )
         except:
             principalComponentTransformer = None

         return arousalPredictor, valencePredictor, principalComponentTransformer

    def __doPrediction( self, features ):
        arousalPredictor, valencePredictor, principalComponentTransformer = self.__loadPredictor()        
        if arousalPredictor is not None and valencePredictor is not None:
            if principalComponentTransformer is not None:
                features = principalComponentTransformer.transform( features.reshape( 1, -1 ) )
            
            arousalPrediction = arousalPredictor.predict( features )[ 0 ]
            valencePrediction = valencePredictor.predict( features )[ 0 ]
            
            return arousalPrediction.item(), valencePrediction.item()

        else:
            return float('nan'), float('nan')

    def post( self ):
        arousalPredictions = []
        valencePredictions = []
        couldNotPredict = False
        recordingID = request.get_json()[ 'recordingID' ]
        requestFeatureVectors = request.get_json()[ 'mfccVectors' ]

        if requestFeatureVectors is not None:
            for requestFeatureVector in requestFeatureVectors:
                #featuresArray = self.__dictionaryToNumpyArray( requestFeatureVector )[ 2 : ]
                featuresArray = np.array( requestFeatureVector[ 'features' ] ).reshape( 1, -1 )
                arousalPrediction, valencePrediction = self.__doPrediction( featuresArray )
                arousalPredictions.append( arousalPrediction )
                valencePredictions.append( valencePrediction )

                if math.isnan( arousalPrediction ) or math.isnan( valencePrediction ):
                    couldNotPredict = True
                else:
                    self.__storeInformation( recordingID, requestFeatureVector, arousalPrediction, valencePrediction )

            predictionsDict = { 'recordingID': recordingID, 'arousal': np.mean( arousalPredictions ), 'valence': np.mean( valencePredictions ) }
            responseJson = jsonify( predictionsDict )               
            if couldNotPredict == True:
                responseJson.status_code = 500
            return responseJson

        else:
            responseJson = jsonify( {} )
            responseJson.status_code = 500
            return responseJson

api = Api( app )
api.add_resource( PredictorAPI, '/predict' )

from predictormanager import routes
