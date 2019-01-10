package groept.be.emodetect;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ListView;

import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

import com.google.gson.Gson;

import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.miscellaneous.ExceptionHandler;
import groept.be.emodetect.serviceclients.SimpleWebServiceClientPost;
import groept.be.emodetect.serviceclients.WebServiceResultHandler;
import groept.be.emodetect.serviceclients.dtos.PredictRequestFactory;
import groept.be.emodetect.serviceclients.dtos.PredictResult;
import groept.be.emodetect.uihelpers.PredictResultListAdapter;

public class PredictionResultsActivity extends AppCompatActivity implements WebServiceResultHandler, ExceptionHandler {
    public static final String PREDICTION_RESULTS_ACTIVITY_TAG = "PredictionResultsActivity";

    ListView predictionResultsListView;
    PredictResultListAdapter predictResultListAdapter;

    String backendWebServiceURL = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_prediction_results);

        backendWebServiceURL = SettingsActivity.getWebServiceURLFromPreferences( getApplicationContext() );

        Intent intent = getIntent();
        ArrayList< String > recordingsToPredict = intent.getStringArrayListExtra( "RECORDINGS_TO_PREDICT" );

        this.predictionResultsListView = ( ListView )( findViewById( R.id.prediction_results_list ) );
        this.predictResultListAdapter = new PredictResultListAdapter(
            this,
            this,
            RecordingsDatabaseHelper.getInstance( getApplicationContext() )
        );
        this.predictionResultsListView.setAdapter( this.predictResultListAdapter );

        URL webServiceURL = null;
        try {
            //webServiceURL = new URL("http://142.93.133.60/predict");
            Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, "WEB SERVICE URL = " + backendWebServiceURL );
            webServiceURL = new URL( backendWebServiceURL );
        } catch( MalformedURLException e ){
            Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, "MalformedURLException occured with message " + e.getMessage());
        }

        for( String currentRecordingToPredict : recordingsToPredict ){
            String currentPredictRequest = PredictRequestFactory.getPredictRequestJSON( getApplicationContext(), currentRecordingToPredict );
            Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, "Executing predict request\n" + currentPredictRequest );

            SimpleWebServiceClientPost webServicePostClient = new SimpleWebServiceClientPost( webServiceURL, this );
            webServicePostClient.execute( currentPredictRequest );
        };
    }

    @Override
    public void handleProperResult( String returnData ){
        Log.d(PREDICTION_RESULTS_ACTIVITY_TAG,"Web Service call result: " + returnData );

        Gson deserializer = new Gson();
        PredictResult thePredictResult = deserializer.fromJson( returnData, PredictResult.class );
        this.predictResultListAdapter.addPredictResult( thePredictResult );
        Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, "Object, recording ID = " + thePredictResult.getRecordingID() + ", arousal = " + thePredictResult.getArousal() + ", valence = " + thePredictResult.getValence() );
    }

    @Override
    public void handleException( Exception exception ){
        Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, "Received exception with message: " + exception.getMessage() );
        Log.d( PREDICTION_RESULTS_ACTIVITY_TAG, exception.getCause().toString() );
    }
}
