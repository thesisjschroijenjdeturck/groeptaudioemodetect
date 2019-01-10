package groept.be.emodetect.uihelpers;

import java.util.ArrayList;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;

import groept.be.emodetect.R;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.miscellaneous.ExceptionHandler;
import groept.be.emodetect.serviceclients.dtos.PredictResult;

public class PredictResultListAdapter extends BaseAdapter {
    private static class ViewHolder{
        private TextView predictedRecordingNameTextView;
        private TextView predictedArousalTextView;
        private TextView predictedValenceTextView;

        public TextView getPredictedRecordingNameTextView(){
            return( this.predictedRecordingNameTextView );
        }

        public void setPredictedRecordingNameTextView( TextView predictedRecordingNameTextView){
            this.predictedRecordingNameTextView = predictedRecordingNameTextView;
        }

        public TextView getPredictedArousalTextView(){
            return( this.predictedArousalTextView );
        }

        public void setPredictedArousalTextView( TextView predictedArousalTextView){
            this.predictedArousalTextView = predictedArousalTextView;
        }

        public TextView getPredictedValenceTextView(){
            return( this.predictedValenceTextView );
        }

        public void setPredictedValenceTextView( TextView predictedValenceTextView){
            this.predictedValenceTextView = predictedValenceTextView;
        }
    }

    public static final String PREDICT_RESULT_LIST_ADAPTER_TAG = "PredictResultListAdapter";

    private Context context;
    private LayoutInflater layoutInflater;
    private ExceptionHandler exceptionHandler;
    private RecordingsDatabaseHelper recordingsDatabaseHelper;

    private ArrayList< PredictResult > predictResults;

    public PredictResultListAdapter(
            Context context,
            ExceptionHandler exceptionHandler,
            RecordingsDatabaseHelper recordingsDatabaseHelper){
        this.context = context;
        this.exceptionHandler = exceptionHandler;
        this.recordingsDatabaseHelper = recordingsDatabaseHelper;

        layoutInflater = ( LayoutInflater ) context.getSystemService( Context.LAYOUT_INFLATER_SERVICE );

        this.predictResults = new ArrayList< PredictResult >();
    }

    @Override
    public int getCount(){
        return( predictResults.size() );
    }

    @Override
    public Object getItem( int position ){
        return( predictResults.get( position ) );
    }

    @Override
    public long getItemId( int position ) {
        return( position );
    }

    @Override
    public View getView( int position, View convertView, ViewGroup parent ){
        View newItemView;
        PredictResultListAdapter.ViewHolder newItemViewSubviews;

        if( convertView == null ){
            newItemView = layoutInflater.inflate( R.layout.predict_result_list_item, parent, false );

            TextView predictedRecordingNameTextView =
                ( TextView )( newItemView.findViewById( R.id.predicted_recording_name ) );
            TextView predictedArousalTextView =
                ( TextView )( newItemView.findViewById( R.id.predicted_arousal_view ) );
            TextView predictedValenceTextView =
                ( TextView )( newItemView.findViewById( R.id.predicted_valence_view ) );

            newItemViewSubviews = new PredictResultListAdapter.ViewHolder();
            newItemViewSubviews.setPredictedRecordingNameTextView(
                predictedRecordingNameTextView
            );
            newItemViewSubviews.setPredictedArousalTextView(
                predictedArousalTextView
            );
            newItemViewSubviews.setPredictedValenceTextView(
                predictedValenceTextView
            );

            newItemView.setTag( newItemViewSubviews );
        } else {
            newItemView = convertView;

            newItemViewSubviews =
                ( PredictResultListAdapter.ViewHolder )( convertView.getTag() );
        }

        newItemViewSubviews.
            getPredictedRecordingNameTextView().
                setText(
                    "Recording name: " +
                    this.recordingsDatabaseHelper.getRecordingFileName(
                        this.predictResults.get( position ).getRecordingID()
                    )
                );
        newItemViewSubviews.
            getPredictedArousalTextView().
                setText(
                    "Arousal: " +
                    this.predictResults.get( position ).getArousal()
                );
        newItemViewSubviews.
            getPredictedValenceTextView().
                setText(
                    "Valence: " +
                    this.predictResults.get( position ).getValence()
                );

        return( newItemView );
    }

    @Override
    public void notifyDataSetChanged(){
        super.notifyDataSetChanged();
    }

    public void addPredictResult( PredictResult predictResultToAdd ){
        this.predictResults.add( predictResultToAdd );

        this.notifyDataSetChanged();
    }

    public void removePredictedResult( PredictResult predictResultToRemove ){
        this.predictResults.remove( predictResultToRemove );

        this.notifyDataSetChanged();
    }
}
