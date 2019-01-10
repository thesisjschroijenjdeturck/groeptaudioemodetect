package groept.be.emodetect.fragments;

import android.arch.lifecycle.ViewModelProviders;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v7.app.AlertDialog;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.Button;

import java.util.ArrayList;
import java.io.IOException;

import groept.be.emodetect.PredictionResultsActivity;
import groept.be.emodetect.R;
import groept.be.emodetect.fragments.models.TestAnalyzeTabModel;
import groept.be.emodetect.uihelpers.dialogs.RecordingLabelingDialog;

public class TestAnalyzeTabFragment extends Fragment {
    private Context activityContext;

    private TestAnalyzeTabModel testAnalyzeTabModel;

    private ListView selectAnalyzableRecordingsListView;
    private Button analyzeSelectedRecordingsButton;

    private RecordingLabelingDialog featureLabelingDialog;

    @Override
    public View onCreateView( LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState ){
        View rootView = inflater.inflate( R.layout.fragment_analyze_tab, null );

        this.activityContext = getActivity();

        this.testAnalyzeTabModel = ViewModelProviders.of( this ).get( TestAnalyzeTabModel.class );

        selectAnalyzableRecordingsListView =
            ( ListView )( rootView.findViewById( R.id.select_analyzable_audio_recordings ) );
        selectAnalyzableRecordingsListView.setAdapter( testAnalyzeTabModel.getSelectRecordingsAdapter() );

        final TestAnalyzeTabModel testAnalyzeTabModelReference =
            this.testAnalyzeTabModel;
        this.analyzeSelectedRecordingsButton =
            rootView.findViewById( R.id.analyze_selected_recordings_button );
        this.analyzeSelectedRecordingsButton.setOnClickListener( new View.OnClickListener(){
            @Override
            public void onClick( View view ){
                /*
                TestAnalyzeTabFragment.this.featureLabelingDialog =
                    new RecordingLabelingDialog(
                        TestAnalyzeTabFragment.this.activityContext
                    );
                TestAnalyzeTabFragment.this.featureLabelingDialog.showDialog();
                */

                ArrayList< String > recordingsToAnalyze =
                    testAnalyzeTabModelReference.
                        getSelectRecordingsAdapter().
                            getSelectedRecordings();

                try {
                    testAnalyzeTabModelReference.
                        getStoredFeatureAnalysisManager().
                        extractFeaturesFromFiles(recordingsToAnalyze);
                } catch( IOException e ){
                    AlertDialog ioErrorDialog =
                        ( new AlertDialog.Builder( TestAnalyzeTabFragment.this.activityContext ) )
                        .create();
                    ioErrorDialog.setTitle("A storage error occured");
                    ioErrorDialog.setMessage( ( "An error occured while writing the extracted feature data:\n" + e.getMessage() ) );
                    ioErrorDialog.setButton( AlertDialog.BUTTON_NEUTRAL, "OK",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int which) {
                                dialog.dismiss();
                            }
                        } );
                    ioErrorDialog.show();
                }

                Intent predictionResultsActivityIntent = new Intent( getActivity(), PredictionResultsActivity.class );
                predictionResultsActivityIntent.putStringArrayListExtra( "RECORDINGS_TO_PREDICT", recordingsToAnalyze );
                startActivity( predictionResultsActivityIntent );
            }
        } );

        return( rootView );
    }
}
