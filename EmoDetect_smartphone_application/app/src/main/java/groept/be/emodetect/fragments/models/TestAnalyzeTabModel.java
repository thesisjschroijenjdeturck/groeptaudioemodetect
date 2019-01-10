package groept.be.emodetect.fragments.models;

import android.app.Application;
import android.arch.lifecycle.AndroidViewModel;
import android.content.Context;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.IOException;

import groept.be.emodetect.helpers.analysishelpers.FeatureAnalysisManager;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.miscellaneous.ExceptionHandler;
import groept.be.emodetect.uihelpers.SelectRecordingsListAdapter;

public class TestAnalyzeTabModel extends AndroidViewModel implements ExceptionHandler {
    public final static String TEST_ANALYZE_TAB_MODEL_TAG = "TestAnalyzeTabModel";

    private static TestAnalyzeTabModel currentInstance = null;
    public static TestAnalyzeTabModel getCurrentInstance(){
        return( TestAnalyzeTabModel.currentInstance );
    }

    private Context applicationContext;

    private FeatureAnalysisManager storedFeatureAnalysisManager;
    private SelectRecordingsListAdapter selectRecordingsListAdapter;
    private RecordingsDatabaseHelper recordingsDatabaseHelper;

    public TestAnalyzeTabModel( Application application ){
        super( application );

        TestAnalyzeTabModel.currentInstance = this;

        this.applicationContext = application;

        this.storedFeatureAnalysisManager =
            new FeatureAnalysisManager( applicationContext );

        this.recordingsDatabaseHelper =
            RecordingsDatabaseHelper.getInstance( applicationContext );

        this.selectRecordingsListAdapter =
            new SelectRecordingsListAdapter(
                applicationContext,
                this,
                this.recordingsDatabaseHelper
            );

        this.recordingsDatabaseHelper.addRecordingsDatabaseObserver( this.selectRecordingsListAdapter );
    }

    public FeatureAnalysisManager getStoredFeatureAnalysisManager(){
        return( this.storedFeatureAnalysisManager );
    }

    public SelectRecordingsListAdapter getSelectRecordingsAdapter(){
        return( this.selectRecordingsListAdapter );
    }

    public RecordingsDatabaseHelper getRecordingsDatabaseHelper() {
        return( this.recordingsDatabaseHelper );
    }

    @Override
    public void handleException( Exception exception ){
        Log.v(
            TEST_ANALYZE_TAB_MODEL_TAG,
            ( "FATAL EXCEPTION CAUSED PROGRAM TO TERMINATE!\n" +
              exception.getMessage() )
        );

        System.exit( 2 );
    }
}
