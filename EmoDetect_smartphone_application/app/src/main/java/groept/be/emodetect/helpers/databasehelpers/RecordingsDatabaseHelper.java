package groept.be.emodetect.helpers.databasehelpers;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.util.Log;

import java.util.ArrayList;
import java.util.UUID;

public class RecordingsDatabaseHelper extends SQLiteOpenHelper {
    private static final String RECORDINGS_DATABASE_HELPER_TAG = "RecordingsDBHelper";
    private static final String RECORDINGS_DATABASE_FILE_NAME = "emotion_detection_database.db";

    private static RecordingsDatabaseHelper firstInstance = null;

    private ArrayList< RecordingsDatabaseObserver > observers;

    private RecordingsDatabaseHelper( Context applicationContext, String recordingsDatabaseFileName ){
        super( applicationContext, recordingsDatabaseFileName, null, 1 );

        this.observers = new ArrayList< RecordingsDatabaseObserver >();
    }

    public static RecordingsDatabaseHelper getInstance( Context applicationContext ){
        if( firstInstance == null ){
            synchronized( RecordingsDatabaseHelper.class ){
                if (firstInstance == null) {
                    firstInstance = new RecordingsDatabaseHelper( applicationContext, RecordingsDatabaseHelper.RECORDINGS_DATABASE_FILE_NAME );
                }
            }
        }

        firstInstance.createTableIfNeeded();

        return firstInstance;
    }

    private boolean checkIfTableExists(){
        SQLiteDatabase recordingsDatabase = this.getReadableDatabase();
        String checkIfTableExistsQuery =
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = '" +
                        RecordingsDatabaseContract.Recordings.TABLE_NAME +
                        "'";
        Cursor checkIfTableExistsResultCursor = recordingsDatabase.rawQuery( checkIfTableExistsQuery, null );

        return( ( checkIfTableExistsResultCursor.getCount() > 0 ) );
    }
    private void createTableIfNeeded(){
        if( !checkIfTableExists() ) {
            /* Here we generate and execute the SQL query that creates our table
             * for keeping track of kept recordings
             */
            SQLiteDatabase recordingsDatabase = this.getWritableDatabase();
            String recordingsTableCreationQuery = "CREATE TABLE " +
                    RecordingsDatabaseContract.Recordings.TABLE_NAME +
                    " ( " +
                    RecordingsDatabaseContract.Recordings.COLUMN_NAME_ID +
                    " TEXT PRIMARY KEY, " +
                    RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME +
                    " TEXT, CONSTRAINT recordingFileNameUnique UNIQUE ( " +
                    RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME +
                    " ) )";
            recordingsDatabase.execSQL( recordingsTableCreationQuery );
        }
    }

    @Override
    public void onCreate( SQLiteDatabase recordingsDatabase ) {
    }

    @Override
    public void onUpgrade( SQLiteDatabase recordingsDatabase, int oldVersion, int newVersion ){
        /* We will never need to migrate data to newer database schemas here */
    }

    private void notifyObservers(){
        for( RecordingsDatabaseObserver currentRecordingsDatabaseObserver : observers ) {
             currentRecordingsDatabaseObserver.notifyRecordingsDatabaseChanged();
        }
    }

    public void addRecordingsDatabaseObserver( RecordingsDatabaseObserver newRecordingsDatabaseObserver ){
        this.observers.add( newRecordingsDatabaseObserver );
    }

    public void removeRecordingsDatabaseObserver( RecordingsDatabaseObserver recordingsDatabaseObserverToRemove ){
        this.observers.remove( recordingsDatabaseObserverToRemove );
    }

    public ArrayList< String > getKeptRecordings(){
        SQLiteDatabase recordingsDatabase = this.getReadableDatabase();
        ArrayList< String > keptRecordings = new ArrayList<>();
        String getKeptRecordingsQuery = "SELECT * FROM " +
                                        RecordingsDatabaseContract.Recordings.TABLE_NAME;
        Cursor keptRecordingsCursor = recordingsDatabase.rawQuery( getKeptRecordingsQuery, null );

        if( keptRecordingsCursor.moveToFirst() ){
            do{
                keptRecordings.add( keptRecordingsCursor.getString( 1 ) );
            } while( keptRecordingsCursor.moveToNext() );
        }

        return( keptRecordings );
    }

    public String getRecordingID( String recordingFileName ){
        String recordingID = null;

        SQLiteDatabase recordingsDatabase = this.getReadableDatabase();
        String getRecordingIDQuery =
            "SELECT ID FROM " +
            RecordingsDatabaseContract.Recordings.TABLE_NAME +
            " WHERE RecordingFileName = '" +
            recordingFileName +
            "';";
        Cursor recordingIDCursor = recordingsDatabase.rawQuery( getRecordingIDQuery, null );

        if( recordingIDCursor.moveToFirst() ){
            recordingID = recordingIDCursor.getString( 0 );
        }

        return( recordingID );
    }

    public String getRecordingFileName( String recordingID ){
        String recordingFileName = null;

        SQLiteDatabase recordingsDatabase = this.getReadableDatabase();
        String getRecordingFileNameQuery =
            "SELECT " +
            RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME +
            " FROM " +
            RecordingsDatabaseContract.Recordings.TABLE_NAME +
            " WHERE " +
            RecordingsDatabaseContract.Recordings.COLUMN_NAME_ID +
            " = '" +
            recordingID +
            "';";
        Cursor recordingFileNameCursor = recordingsDatabase.rawQuery( getRecordingFileNameQuery, null );

        if( recordingFileNameCursor.moveToFirst() ){
            recordingFileName = recordingFileNameCursor.getString( 0 );
        }

        return( recordingFileName );
    }

    public boolean isRecorded( String recordingFileName ){
        SQLiteDatabase recordingsDatabase = this.getReadableDatabase();
        String findRecordingQuery = "SELECT * FROM " +
                                    RecordingsDatabaseContract.Recordings.TABLE_NAME +
                                    " WHERE " +
                                    RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME +
                                    " = '" +
                                    recordingFileName +
                                    "'";
        Log.d( RECORDINGS_DATABASE_HELPER_TAG, "Executing query: " + findRecordingQuery );
        Cursor findRecordingResultCursor = recordingsDatabase.rawQuery( findRecordingQuery, null );
        Log.d( RECORDINGS_DATABASE_HELPER_TAG, "# result rows = " + findRecordingResultCursor.getCount() );

        return( ( findRecordingResultCursor.getCount() > 0 ) );
    }

    public void insertRecording( String newRecordingFileName ){
        SQLiteDatabase recordingsDatabase = this.getWritableDatabase();

        ContentValues contentValue = new ContentValues();
        contentValue.put( RecordingsDatabaseContract.Recordings.COLUMN_NAME_ID, UUID.randomUUID().toString() );
        contentValue.put( RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME, newRecordingFileName );

        Log.d( RECORDINGS_DATABASE_HELPER_TAG, "Checking for existence" );
        if( isRecorded( newRecordingFileName ) ){
            Log.d( RECORDINGS_DATABASE_HELPER_TAG, newRecordingFileName + " was already recorded!" );
        }
        if( !isRecorded( newRecordingFileName ) ){
            Log.d( RECORDINGS_DATABASE_HELPER_TAG, newRecordingFileName + " was NOT already recorded!" );
            recordingsDatabase.insert(RecordingsDatabaseContract.Recordings.TABLE_NAME, null, contentValue);
        }

        this.notifyObservers();
    }

    public void deleteRecording( String recordingToDeleteFileName ){
        SQLiteDatabase recordingsDatabase = this.getWritableDatabase();

        String deletionQuery = "DELETE FROM " +
                               RecordingsDatabaseContract.Recordings.TABLE_NAME +
                               " WHERE " +
                               RecordingsDatabaseContract.Recordings.COLUMN_NAME_RECORDING_FILE_NAME +
                               " = '" +
                               recordingToDeleteFileName +
                               "'";

        if( isRecorded( recordingToDeleteFileName ) ){
            recordingsDatabase.execSQL( deletionQuery );
        }

        this.notifyObservers();
    }
}
