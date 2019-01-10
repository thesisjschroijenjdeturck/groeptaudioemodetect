package groept.be.emodetect.helpers.databasehelpers.featuresdatabases;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.util.Log;
import android.support.v4.util.Pair;

import java.util.ArrayList;

import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseContract;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.serviceclients.dtos.FeatureVector;

public class MfccsDatabaseHelper extends FeaturesDatabaseHelper {
    private static final String MFCC_FEATURES_DATABASE_HELPER_TAG = "MfccFeaturesDBHelper";
    private static final String MFCC_FEATURES_DATABASE_FILE_NAME = "emotion_detection_database.db";

    private static MfccsDatabaseHelper firstInstance = null;

    private Context applicationContext;

    private MfccsDatabaseHelper( Context applicationContext, String featuresDatabaseFileName ) {
        super( applicationContext, featuresDatabaseFileName );

        this.applicationContext = applicationContext;
    }

    public static MfccsDatabaseHelper getInstance( Context applicationContext ){
        if( firstInstance == null ){
            synchronized( MfccsDatabaseHelper.class ){
                if( firstInstance == null ){
                    firstInstance = new MfccsDatabaseHelper( applicationContext, MfccsDatabaseHelper.MFCC_FEATURES_DATABASE_FILE_NAME );
                }
            }
        }

        firstInstance.createTableIfNeeded();

        return firstInstance;
    }

    private Long insertFeature( SQLiteDatabase mfccsDatabase, String recordingID, int frameOffset, float[] newFeature ){
        ContentValues contentValues = new ContentValues();
        contentValues.put( getContractFieldValue( "COLUMN_NAME_RECORDING_ID" ), recordingID );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_FRAME_OFFSET" ), frameOffset );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_0" ), newFeature[ 0 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_1" ), newFeature[ 1 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_2" ), newFeature[ 2 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_3" ), newFeature[ 3 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_4" ), newFeature[ 4 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_5" ), newFeature[ 5 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_6" ), newFeature[ 6 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_7" ), newFeature[ 7 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_8" ), newFeature[ 8 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_9" ), newFeature[ 9 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_10" ), newFeature[ 10 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_11" ), newFeature[ 11 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_COEFFICIENT_12" ), newFeature[ 12 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_0" ), newFeature[ 13 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_1" ), newFeature[ 14 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_2" ), newFeature[ 15 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_3" ), newFeature[ 16 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_4" ), newFeature[ 17 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_5" ), newFeature[ 18 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_6" ), newFeature[ 19 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_7" ), newFeature[ 20 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_8" ), newFeature[ 21 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_9" ), newFeature[ 22 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_10" ), newFeature[ 23 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_11" ), newFeature[ 24 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_12" ), newFeature[ 25 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_0" ), newFeature[ 26 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_1" ), newFeature[ 27 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_2" ), newFeature[ 28 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_3" ), newFeature[ 29 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_4" ), newFeature[ 30 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_5" ), newFeature[ 31 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_6" ), newFeature[ 32 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_7" ), newFeature[ 33 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_8" ), newFeature[ 34 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_9" ), newFeature[ 35 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_10" ), newFeature[ 36 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_11" ), newFeature[ 37 ] );
        contentValues.put( getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_12" ), newFeature[ 38 ] );

        return( mfccsDatabase.insert( getContractFieldValue( "TABLE_NAME" ), null, contentValues ) );
    }

    protected Class getContractClass(){
        return( MfccsDatabaseContract.class );
    }

    public void createTableIfNeeded(){
        if( !checkIfTableExists() ) {
            /* Here we generate and execute the SQL query that creates our table
             * for storing extracted MFCC features
             */
            SQLiteDatabase mfccsDatabase = this.getWritableDatabase();
            String mfccsTableCreationQuery = "CREATE TABLE " +
                getContractFieldValue("TABLE_NAME") +
                " ( " +
                getContractFieldValue("COLUMN_NAME_ID") +
                " INTEGER PRIMARY KEY, " +
                getContractFieldValue("COLUMN_NAME_RECORDING_ID") +
                " TEXT, " +
                getContractFieldValue("COLUMN_NAME_FRAME_OFFSET") +
                " INTEGER, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_0") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_1") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_2") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_3") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_4") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_5") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_6") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_7") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_8") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_9") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_10") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_11") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_COEFFICIENT_12") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_0") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_1") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_2") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_3") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_4") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_5") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_6") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_7") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_8") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_9") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_10") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_11") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_COEFFICIENT_12") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_0") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_1") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_2") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_3") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_4") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_5") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_6") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_7") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_8") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_9") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_10") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_11") +
                " REAL, " +
                getContractFieldValue("COLUMN_NAME_DELTA_DELTA_COEFFICIENT_12") +
                " REAL, FOREIGN KEY ( " +
                getContractFieldValue("COLUMN_NAME_RECORDING_ID") +
                " ) REFERENCES " +
                RecordingsDatabaseContract.Recordings.TABLE_NAME +
                " ( " +
                RecordingsDatabaseContract.Recordings.COLUMN_NAME_ID +
                " ) )";
            mfccsDatabase.execSQL(mfccsTableCreationQuery);
            Log.d(MFCC_FEATURES_DATABASE_HELPER_TAG, "Executed table creation query: " + mfccsTableCreationQuery);
            Log.d("DEBUG", "Executed table creation query: " + mfccsTableCreationQuery);
        }
    }

    @Override
    public boolean isAnalyzed( String recordingID ){
        SQLiteDatabase mfccsDatabase = this.getReadableDatabase();
        String findRecordingQuery = "SELECT * FROM " +
                                    getContractFieldValue( "TABLE_NAME" ) +
                                    " WHERE " +
                                    getContractFieldValue( "COLUMN_NAME_RECORDING_ID" ) +
                                    " = '" +
                                    recordingID +
                                    "'";
        Cursor findRecordingResultCursor = mfccsDatabase.rawQuery( findRecordingQuery, null );

        return( ( findRecordingResultCursor.getCount() > 0 ) );
    }

    @Override
    public boolean insertFeatures( String recordingID, ArrayList< Pair< Integer, float[] > > featuresToStore ){
        if( this.isAnalyzed( recordingID ) ){
            this.removeFeatures( recordingID );
        }

        SQLiteDatabase mfccsDatabase = this.getWritableDatabase();

        boolean allInsertsSuccessful = true;
        try{
            mfccsDatabase.beginTransaction();

            for( Pair< Integer, float[] > currentFeatureToInsert : featuresToStore ){
                long insertResult = insertFeature( mfccsDatabase, recordingID, currentFeatureToInsert.first, currentFeatureToInsert.second );

                if( insertResult == -1 ){
                    allInsertsSuccessful = false;
                    break;
                }
            }
        } catch( Exception e ){
        } finally{
            if( allInsertsSuccessful == true ){
                mfccsDatabase.setTransactionSuccessful();
            }
            mfccsDatabase.endTransaction();

            return( allInsertsSuccessful );
        }
    }

    @Override
    public void removeFeatures( String recordingID ){
        SQLiteDatabase mfccsDatabase = this.getWritableDatabase();

        mfccsDatabase.delete(
            getContractFieldValue( "TABLE_NAME" ),
            ( getContractFieldValue( "COLUMN_NAME_RECORDING_ID" ) + " = ?" ),
            new String[]{ recordingID }
        );
    }

    // @Override
    public ArrayList< FeatureVector > getFeatureVectorList( String recordingFilename ){
        ArrayList< FeatureVector > featureVectorList = null;

        RecordingsDatabaseHelper recordingsDatabaseHelper = RecordingsDatabaseHelper.getInstance( this.applicationContext );
        String recordingId = recordingsDatabaseHelper.getRecordingID( recordingFilename );

        SQLiteDatabase mfccsDatabase = this.getReadableDatabase();
        String featureVectorsQuery =
                "SELECT " +
                getContractFieldValue( "COLUMN_NAME_FRAME_OFFSET" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_0" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_1" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_2" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_3" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_4" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_5" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_6" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_7" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_8" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_9" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_10" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_11" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_COEFFICIENT_12" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_0" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_1" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_2" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_3" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_4" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_5" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_6" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_7" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_8" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_9" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_10" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_11" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_COEFFICIENT_12" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_0" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_1" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_2" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_3" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_4" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_5" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_6" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_7" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_8" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_9" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_10" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_11" ) + ", " +
                getContractFieldValue( "COLUMN_NAME_DELTA_DELTA_COEFFICIENT_12" ) +

                " FROM " +
                getContractFieldValue( "TABLE_NAME" ) +
                " WHERE " +
                getContractFieldValue( "COLUMN_NAME_RECORDING_ID" ) +
                " = '" +
                recordingId +
                "' ORDER BY " +
                getContractFieldValue( "COLUMN_NAME_FRAME_OFFSET" ) +
                " ASC";

        Log.d( MFCC_FEATURES_DATABASE_HELPER_TAG, "Executing query: " + featureVectorsQuery );
        Cursor featureVectorsCursor = mfccsDatabase.rawQuery( featureVectorsQuery, null );
        Log.d( MFCC_FEATURES_DATABASE_HELPER_TAG, "# result rows = " + featureVectorsCursor.getCount() );

        if( featureVectorsCursor.moveToFirst() ){
            featureVectorList = new ArrayList< FeatureVector >();

            do {
                int currentFrameOffset;
                float[] currentMfccFeatures = new float[ 39 ];

                currentFrameOffset = featureVectorsCursor.getInt( 0 );
                currentMfccFeatures[ 0 ] = featureVectorsCursor.getFloat( 1 );
                currentMfccFeatures[ 1 ] = featureVectorsCursor.getFloat( 2 );
                currentMfccFeatures[ 2 ] = featureVectorsCursor.getFloat( 3 );
                currentMfccFeatures[ 3 ] = featureVectorsCursor.getFloat( 4 );
                currentMfccFeatures[ 4 ] = featureVectorsCursor.getFloat( 5 );
                currentMfccFeatures[ 5 ] = featureVectorsCursor.getFloat( 6 );
                currentMfccFeatures[ 6 ] = featureVectorsCursor.getFloat( 7 );
                currentMfccFeatures[ 7 ] = featureVectorsCursor.getFloat( 8 );
                currentMfccFeatures[ 8 ] = featureVectorsCursor.getFloat( 9 );
                currentMfccFeatures[ 9 ] = featureVectorsCursor.getFloat( 10 );
                currentMfccFeatures[ 10 ] = featureVectorsCursor.getFloat( 11 );
                currentMfccFeatures[ 11 ] = featureVectorsCursor.getFloat( 12 );
                currentMfccFeatures[ 12 ] = featureVectorsCursor.getFloat( 13 );
                currentMfccFeatures[ 13 ] = featureVectorsCursor.getFloat( 14 );
                currentMfccFeatures[ 14 ] = featureVectorsCursor.getFloat( 15 );
                currentMfccFeatures[ 15 ] = featureVectorsCursor.getFloat( 16 );
                currentMfccFeatures[ 16 ] = featureVectorsCursor.getFloat( 17 );
                currentMfccFeatures[ 17 ] = featureVectorsCursor.getFloat( 18 );
                currentMfccFeatures[ 18 ] = featureVectorsCursor.getFloat( 19 );
                currentMfccFeatures[ 19 ] = featureVectorsCursor.getFloat( 20 );
                currentMfccFeatures[ 20 ] = featureVectorsCursor.getFloat( 21 );
                currentMfccFeatures[ 21 ] = featureVectorsCursor.getFloat( 22 );
                currentMfccFeatures[ 22 ] = featureVectorsCursor.getFloat( 23 );
                currentMfccFeatures[ 23 ] = featureVectorsCursor.getFloat( 24 );
                currentMfccFeatures[ 24 ] = featureVectorsCursor.getFloat( 25 );
                currentMfccFeatures[ 25 ] = featureVectorsCursor.getFloat( 26 );
                currentMfccFeatures[ 26 ] = featureVectorsCursor.getFloat( 27 );
                currentMfccFeatures[ 27 ] = featureVectorsCursor.getFloat( 28 );
                currentMfccFeatures[ 28 ] = featureVectorsCursor.getFloat( 29 );
                currentMfccFeatures[ 29 ] = featureVectorsCursor.getFloat( 30 );
                currentMfccFeatures[ 30 ] = featureVectorsCursor.getFloat( 31 );
                currentMfccFeatures[ 31 ] = featureVectorsCursor.getFloat( 32 );
                currentMfccFeatures[ 32 ] = featureVectorsCursor.getFloat( 33 );
                currentMfccFeatures[ 33 ] = featureVectorsCursor.getFloat( 34 );
                currentMfccFeatures[ 34 ] = featureVectorsCursor.getFloat( 35 );
                currentMfccFeatures[ 35 ] = featureVectorsCursor.getFloat( 36 );
                currentMfccFeatures[ 36 ] = featureVectorsCursor.getFloat( 37 );
                currentMfccFeatures[ 37 ] = featureVectorsCursor.getFloat( 38 );
                currentMfccFeatures[ 38 ] = featureVectorsCursor.getFloat( 39 );

                featureVectorList.add( new FeatureVector( currentFrameOffset, currentMfccFeatures ) );
            } while( featureVectorsCursor.moveToNext() );
        }

        return( featureVectorList );
    }
}
