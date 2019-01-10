package groept.be.emodetect.serviceclients.dtos;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;

import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.databasehelpers.featuresdatabases.MfccsDatabaseHelper;

public class PredictRequestFactory {
    public static String getPredictRequestJSON( Context applicationContext, String recordingFilename ){
        Gson jsonSerializer = new Gson();

        RecordingsDatabaseHelper recordingsDatabaseHelper = RecordingsDatabaseHelper.getInstance( applicationContext );
        MfccsDatabaseHelper mfccsDatabaseHelper = MfccsDatabaseHelper.getInstance( applicationContext );

        RecordingFeatureVectors recordingFeatureVectors =
            new RecordingFeatureVectors(
                recordingsDatabaseHelper.getRecordingID( recordingFilename ),
                mfccsDatabaseHelper.getFeatureVectorList( recordingFilename )
            );

        return( jsonSerializer.toJson( recordingFeatureVectors ) );
    }
}
