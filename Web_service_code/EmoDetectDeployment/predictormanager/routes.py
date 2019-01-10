from os import path

from flask import render_template, flash, redirect, url_for

from predictormanager import app, db
import predictormanager.models

# predictors = [ ( i.predictorName, i.predictorName ) for i in predictormanager.models.Predictor.query.all() ]

from predictormanager.forms import PredictorSelectorForm, PredictorUploadForm

@app.route( '/', methods = [ 'GET', 'POST' ] )
def index():
    predictorSelectorForm = PredictorSelectorForm()
    predictorUploadForm = PredictorUploadForm()
    predictors = [ ( i.predictorName, i.predictorName ) for i in predictormanager.models.Predictor.query.all() ]
    predictorSelectorForm.predictor.choices = predictors

    if predictorSelectorForm.validate_on_submit():
        oldPredictorRow = predictormanager.models.Predictor.query.filter_by( active = True )
        oldPredictorRow.active = False
        selectedPredictor = predictorSelectorForm.predictor.data
        selectedPredictorRow = predictormanager.models.Predictor.query.filter_by( predictorName = selectedPredictor ).first()
        selectedPredictorRow.active = True
        db.session.commit()

    if predictorUploadForm.validate_on_submit():
        newArousalPredictorFilename = predictorUploadForm.arousalPredictorFileSelector.data.filename
        newValencePredictorFilename = predictorUploadForm.valencePredictorFileSelector.data.filename
        newArousalPredictorFilePath = path.join( app.root_path, "uploaded_predictors", "arousal", newArousalPredictorFilename )
        newValencePredictorFilePath = path.join( app.root_path, "uploaded_predictors", "valence", newValencePredictorFilename )
        predictorUploadForm.arousalPredictorFileSelector.data.save( newArousalPredictorFilePath )
        predictorUploadForm.valencePredictorFileSelector.data.save( newValencePredictorFilePath )
        
        if predictorUploadForm.principalComponentTransformerFileSelector.data is not None:
           newPrincipalComponentTransformerFilename = predictorUploadForm.principalComponentTransformerFileSelector.data.filename
           newPrincipalComponentTransformerFilePath = path.join( app.root_path, "uploaded_predictors", "principal_component_transformers", newPrincipalComponentTransformerFilename )
           predictorUploadForm.principalComponentTransformerFileSelector.data.save( newPrincipalComponentTransformerFilePath )
        else:
           newPrincipalComponentTransformerFilename = ""

        newPredictor = predictormanager.models.Predictor()
        newPredictor.predictorName = predictorUploadForm.predictorName.data
        newPredictor.arousalPredictorFilename = newArousalPredictorFilename
        newPredictor.valencePredictorFilename = newValencePredictorFilename
        newPredictor.principalComponentTransformerFilename = newPrincipalComponentTransformerFilename
        if predictormanager.models.Predictor.query.count() < 1:
          newPredictor.active = True
        else:
          newPredictor.active = False

        db.session.add( newPredictor )
        db.session.commit()

    return render_template( "index.html", predictorSelectorForm = predictorSelectorForm, predictorUploadForm = predictorUploadForm )
