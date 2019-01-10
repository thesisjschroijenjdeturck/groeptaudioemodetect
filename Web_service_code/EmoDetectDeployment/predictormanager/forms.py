from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, RadioField, SubmitField
from wtforms.validators import DataRequired

#from predictormanager.routes import predictors

class PredictorSelectorForm( FlaskForm ):
#   predictor = RadioField( 'TEST', choices = predictors )
    predictor = RadioField( 'SELECT THE PREDICTOR TO USE' )

class PredictorUploadForm( FlaskForm ):
    predictorName = StringField( 'New predictor name', validators = [ DataRequired() ] )
    arousalPredictorFileSelector = FileField( 'Arousal predictor pickle file', validators = [ FileRequired(), FileAllowed( [ 'pickle' ] ) ] )
    valencePredictorFileSelector = FileField( 'Valence predictor pickle file', validators = [ FileRequired(), FileAllowed( [ 'pickle' ] ) ] )
    principalComponentTransformerFileSelector = FileField( 'Principal component pickle file', validators = [ FileAllowed( [ 'pickle' ] ) ] )
    submit = SubmitField( 'Upload' )
