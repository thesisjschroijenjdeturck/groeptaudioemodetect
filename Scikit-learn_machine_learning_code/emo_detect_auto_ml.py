import os
import numpy as np
from sklearn.externals import joblib

import emo_detect_auto_ml_global_consts as consts
import mod_logger as logger
import mod_data_loading as data_loading
import mod_data_plotting as data_plotting
import mod_features_preprocessing as features_preprocessing
import mod_svr_model_trainer as svr_model_trainer
import mod_mlp_model_trainer as mlp_model_trainer
import mod_lr_model_trainer as lr_model_trainer
import mod_lasso_model_trainer as lasso_model_trainer
import mod_pca as pca
import mod_ridge_model_trainer as ridge_model_trainer
import mod_scorers as scorers
import mod_score_estimator as score_estimator

if __name__ == "__main__":
    featuresTrain, arousalLabelsTrain, valenceLabelsTrain = data_loading.loadData( 'TRAIN', ( consts.develSetSize + 1 ), consts.trainSetSize, 1 )
    featuresTest, arousalLabelsTest, valenceLabelsTest = data_loading.loadData( 'DEVEL', 1, consts.develSetSize, 1 )

    featuresTrain, featuresTest = pca.transformToPrincipalComponents( 0.9, featuresTrain, featuresTest, dumpTransformer = True, dumpedTransformerFilename = "sewa_dataset_principal_component_transformer.pickle" )

    if not os.path.isdir( 'saved_regressors' ):
        os.mkdir( 'saved_regressors' )
    if not os.path.isdir( 'cv_results' ):
        os.mkdir( 'cv_results' )
    
    gridSearchArousalLr, gridSearchValenceLr = lr_model_trainer.trainModel( featuresTrain, arousalLabelsTrain, valenceLabelsTrain )
    logger.logMessage( "MAIN", "--- Grid search for ideal OLR's with parameter preprocessing ---" )
    logger.logMessage( "MAIN", "Grid search arousal OLR's best CV score: " + str( gridSearchArousalLr.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchArousalLr.best_params_ ) + "\n" )
    logger.logMessage( "MAIN", "Grid search valence OLR's best CV score: " + str( gridSearchValenceLr.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchValenceLr.best_params_ ) + "\n" )
    bestArousalLrScore = score_estimator.scoreEstimator( gridSearchArousalLr.best_estimator_, scorers.calculateCcc, featuresTest, arousalLabelsTest )
    bestValenceLrScore = score_estimator.scoreEstimator( gridSearchValenceLr.best_estimator_, scorers.calculateCcc, featuresTest, valenceLabelsTest )
    logger.logMessage( "MAIN", "Best arousal linear regressor CCC score is " + str( bestArousalLrScore ) )
    logger.logMessage( "MAIN", "Best valence linear regressor CCC score is " + str( bestValenceLrScore ) )
    joblib.dump( gridSearchArousalLr.best_estimator_, "saved_regressors/sewa_dataset_lr_arousal.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceLr.best_estimator_, "saved_regressors/sewa_dataset_lr_valence.pickle", protocol = 2 )
    joblib.dump( gridSearchArousalLr.cv_results_, "cv_results/sewa_dataset_lr_arousal_lr_results.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceLr.cv_results_, "cv_results/sewa_dataset_lr_valence_lr_results.pickle", protocol = 2 )

    gridSearchArousalLassoReg, gridSearchValenceLassoReg = lasso_model_trainer.trainModel( featuresTrain, arousalLabelsTrain, valenceLabelsTrain )
    logger.logMessage( "MAIN", "--- Grid search for ideal Lasso Regression's with parameter preprocessing ---" )
    logger.logMessage( "MAIN", "Grid search arousal Lasso Regression's best CV score: " + str( gridSearchArousalLassoReg.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchArousalLassoReg.best_params_ ) + "\n" )
    logger.logMessage( "MAIN", "Grid search valence Lasso Regression's best CV score: " + str( gridSearchValenceLassoReg.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchValenceLassoReg.best_params_ ) + "\n" )
    bestArousalLassoRegScore = score_estimator.scoreEstimator( gridSearchArousalLassoReg.best_estimator_, scorers.calculateCcc, featuresTest, arousalLabelsTest )
    bestValenceLassoRegScore = score_estimator.scoreEstimator( gridSearchValenceLassoReg.best_estimator_, scorers.calculateCcc, featuresTest, valenceLabelsTest )
    logger.logMessage( "MAIN", "Best arousal lasso regressor CCC score is " + str( bestArousalLassoRegScore ) )
    logger.logMessage( "MAIN", "Best valence lasso regressor CCC score is " + str( bestValenceLassoRegScore ) )
    joblib.dump( gridSearchArousalLassoReg.best_estimator_, "saved_regressors/sewa_dataset_lasso_reg_arousal.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceLassoReg.best_estimator_, "saved_regressors/sewa_dataset_lasso_reg_valence.pickle", protocol = 2 )
    joblib.dump( gridSearchArousalLassoReg.cv_results_, "cv_results/sewa_dataset_lasso_reg_arousal_cv_results.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceLassoReg.cv_results_, "cv_results/sewa_dataset_lasso_reg_valence_cv_results.pickle", protocol = 2 )

    gridSearchArousalRidgeReg, gridSearchValenceRidgeReg = ridge_model_trainer.trainModel( featuresTrain, arousalLabelsTrain, valenceLabelsTrain )
    logger.logMessage( "MAIN", "--- Grid search for ideal Ridge Regression's with parameter preprocessing ---" )
    logger.logMessage( "MAIN", "Grid search arousal Ridge Regression's best CV score: " + str( gridSearchArousalRidgeReg.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchArousalRidgeReg.best_params_ ) + "\n" )
    logger.logMessage( "MAIN", "Grid search valence Ridge Regression's best CV score: " + str( gridSearchValenceRidgeReg.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchValenceRidgeReg.best_params_ ) + "\n" )
    bestArousalRidgeRegScore = score_estimator.scoreEstimator( gridSearchArousalRidgeReg.best_estimator_, scorers.calculateCcc, featuresTest, arousalLabelsTest )
    bestValenceRidgeRegScore = score_estimator.scoreEstimator( gridSearchValenceRidgeReg.best_estimator_, scorers.calculateCcc, featuresTest, valenceLabelsTest )
    logger.logMessage( "MAIN", "Best arousal ridge regressor CCC score is " + str( bestArousalRidgeRegScore ) )
    logger.logMessage( "MAIN", "Best valence ridge regressor CCC score is " + str( bestValenceRidgeRegScore ) )
    joblib.dump( gridSearchArousalRidgeReg.best_estimator_, "saved_regressors/sewa_dataset_ridge_reg_arousal.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceRidgeReg.best_estimator_, "saved_regressors/sewa_dataset_ridge_reg_valence.pickle", protocol = 2 )
    joblib.dump( gridSearchArousalRidgeReg.cv_results_, "cv_results/sewa_dataset_ridge_reg_arousal_cv_results.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceRidgeReg.cv_results_, "cv_results/sewa_dataset_ridge_reg_valence_cv_results.pickle", protocol = 2 )

    featuresTrain, arousalLabelsTrain, valenceLabelsTrain = data_loading.loadData( 'TRAIN', ( consts.develSetSize + 1 ), consts.trainSetSize, 0.005 )
    featuresTest, arousalLabelsTest, valenceLabelsTest = data_loading.loadData( 'DEVEL', 1, consts.develSetSize, 1 )   
    featuresTrain, featuresTest = pca.transformToPrincipalComponents( 0.9, featuresTrain, featuresTest )
    featuresTrain, featuresTest = features_preprocessing.preprocessFeaturesSvr( featuresTrain, featuresTest )
    gridSearchArousalSvr, gridSearchValenceSvr = svr_model_trainer.trainModel( featuresTrain, arousalLabelsTrain, valenceLabelsTrain )
    logger.logMessage( "MAIN", "--- Grid search for ideal SVM's with parameter preprocessing ---" )
    logger.logMessage( "MAIN", "Grid search arousal SVR's best CV score: " + str( gridSearchArousalSvr.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchArousalSvr.best_params_ ) + "\n" )
    logger.logMessage( "MAIN", "Grid search valence SVR's best CV score: " + str( gridSearchValenceSvr.best_score_ ) + "\n" )
    logger.logMessage( "MAIN", "Corresponding hyperparameters: " + str( gridSearchValenceSvr.best_params_ ) + "\n" )
    bestArousalSvrScore = score_estimator.scoreEstimator( gridSearchArousalSvr.best_estimator_, scorers.calculateCcc, featuresTest, arousalLabelsTest )
    bestValenceSvrScore = score_estimator.scoreEstimator( gridSearchValenceSvr.best_estimator_, scorers.calculateCcc, featuresTest, valenceLabelsTest )
    logger.logMessage( "MAIN", "Best arousal support vector regressor CCC score is " + str( bestArousalSvrScore ) )
    logger.logMessage( "MAIN", "Best valence support vector regressor CCC score is " + str( bestValenceSvrScore ) )
    joblib.dump( gridSearchArousalSvr.best_estimator_, "saved_regressors/sewa_dataset_svr_arousal.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceSvr.best_estimator_, "saved_regressors/sewa_dataset_svr_valence.pickle", protocol = 2 )
    joblib.dump( gridSearchArousalSvr.cv_results_, "cv_results/sewa_dataset_svr_arousal_cv_results.pickle", protocol = 2 )
    joblib.dump( gridSearchValenceSvr.cv_results_, "cv_results/sewa_dataset_svr_valence_cv_results.pickle", protocol = 2 )