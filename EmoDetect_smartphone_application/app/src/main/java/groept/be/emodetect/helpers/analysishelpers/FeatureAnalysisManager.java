package groept.be.emodetect.helpers.analysishelpers;

import android.os.Environment;
import android.content.Context;
import android.support.v4.util.Pair;
import android.util.Log;

import java.util.ArrayList;
import java.io.IOException;

import be.tarsos.dsp.io.android.AndroidFFMPEGLocator;

import groept.be.emodetect.helpers.databasehelpers.featuresdatabases.MfccsDatabaseHelper;

/* BELANGRIJK OM OVER NA TE DENKEN - INVALIDE STAAT NA SLUITEN? */
public class FeatureAnalysisManager {
    private static final String FEATURE_ANALYSIS_MANAGER_TAG = "FeatureAnalysisManager";
    private static final String RECORDING_FILENAME_PREFIX = Environment.getExternalStorageDirectory().getAbsolutePath() + "/GroepT/SpeechEmotionDetection/";

    private Context applicationContext;

    private ArrayList< FeatureExtractor > featureExtractors;

    public FeatureAnalysisManager( Context applicationContext ){
        this.applicationContext = applicationContext;

        MfccExtractor defaultMfccExtractor =
            new MfccExtractor( applicationContext, 44100, 1024, 128, 13, 50, 300, 3000 );
        this.featureExtractors = new ArrayList< FeatureExtractor >();
        this.addFeatureExtractor( defaultMfccExtractor );

        MfccsDatabaseHelper mfccsDatabaseHelper = MfccsDatabaseHelper.getInstance( applicationContext );

        new AndroidFFMPEGLocator( this.applicationContext );
    }

    public void addFeatureExtractor( FeatureExtractor featureExtractorToAdd ){
        featureExtractors.add( featureExtractorToAdd );
    }

    public void removeFeatureExtractor( FeatureExtractor featureExtractorToRemove ){
        featureExtractors.remove( featureExtractorToRemove );
    }

    public void extractFeaturesFromFile( String recordingFilename ) throws IOException {
        for( FeatureExtractor currentFeatureExtractor : featureExtractors ){
            ArrayList< Pair< Integer, float[] > > extractedFeatures =
                currentFeatureExtractor.extractFeatures( RECORDING_FILENAME_PREFIX + recordingFilename );
        }
    }

    public void extractFeaturesFromFiles( ArrayList<String> recordingFilenames ) throws IOException {
        for( String currentRecordingFilename : recordingFilenames ) {
            extractFeaturesFromFile( currentRecordingFilename );
        }
    }
}
