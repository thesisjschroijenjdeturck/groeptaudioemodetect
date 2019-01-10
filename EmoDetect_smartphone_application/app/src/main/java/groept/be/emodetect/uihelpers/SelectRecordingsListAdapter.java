package groept.be.emodetect.uihelpers;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

import android.content.Context;
import android.util.Pair;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.CompoundButton;
import android.widget.CheckBox;
import android.widget.TextView;

import groept.be.emodetect.R;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseObserver;
import groept.be.emodetect.helpers.miscellaneous.ExceptionHandler;

public class SelectRecordingsListAdapter extends BaseAdapter implements RecordingsDatabaseObserver {
    private static class ViewHolder{
        private TextView selectableRecordingNameView;
        private CheckBox markedAsSelectedRecordingView;

        public TextView getSelectableRecordingNameView(){
            return( this.selectableRecordingNameView);
        }

        public void setSelectableRecordingNameView(TextView selectableRecordingNameView){
            this.selectableRecordingNameView = selectableRecordingNameView;
        }

        public CheckBox getMarkedAsSelectedRecordingView(){
            return( this.markedAsSelectedRecordingView);
        }

        public void setMarkedAsSelectedRecordingView(CheckBox markedAsSelectedRecordingView){
            this.markedAsSelectedRecordingView = markedAsSelectedRecordingView;
        }
    }

    private static class SelectedWatcher implements CompoundButton.OnCheckedChangeListener{
        private AtomicBoolean selectedBooleanReference;

        public SelectedWatcher(AtomicBoolean selectedBooleanReference ){
            this.selectedBooleanReference = selectedBooleanReference;
        }

        @Override
        public void onCheckedChanged( android.widget.CompoundButton compoundButton, boolean b ){
            this.selectedBooleanReference.set( b );
        }
    }

    public static final String ANALYZABLES_LIST_ADAPTER_TAG = "AnalyzablesListAdapter";

    private Context context;
    private LayoutInflater layoutInflater;
    private ExceptionHandler exceptionHandler;
    private RecordingsDatabaseHelper recordingsDatabase;

    private ArrayList< Pair< String, AtomicBoolean > > selectableRecordings;

    public SelectRecordingsListAdapter(
        Context context,
        ExceptionHandler exceptionHandler,
        RecordingsDatabaseHelper recordingsDatabase ){
        this.context = context;
        this.exceptionHandler = exceptionHandler;
        this.recordingsDatabase = recordingsDatabase;

        layoutInflater = ( LayoutInflater ) context.getSystemService( Context.LAYOUT_INFLATER_SERVICE );

        this.getSelectableRecordings();
    }

    private void getSelectableRecordings(){
        this.selectableRecordings = new ArrayList< Pair< String, AtomicBoolean > >();
        ArrayList< String > storedRecordings = recordingsDatabase.getKeptRecordings();
        for( String currentRecording : storedRecordings ){
            this.selectableRecordings.add(
                    new Pair< String, AtomicBoolean >(
                            currentRecording,
                            new AtomicBoolean( false )
                    )
            );
        }
    }

    @Override
    public int getCount(){
        return( selectableRecordings.size() );
    }

    @Override
    public Object getItem( int position ){
        return( selectableRecordings.get( position ).first );
    }

    @Override
    public long getItemId( int position ) {
        return( position );
    }

    @Override
    public View getView( int position, View convertView, ViewGroup parent ){
        View newRowView;
        SelectRecordingsListAdapter.ViewHolder newRowViewSubviews;

        if( convertView == null ){
            newRowView = layoutInflater.inflate( R.layout.select_recordings_list_item, parent, false );

            TextView selectableRecordingNameView =
                ( TextView )( newRowView.findViewById( R.id.selectable_recording_name ) );
            CheckBox markAsSelectedRecordingView =
                ( CheckBox )( newRowView.findViewById( R.id.marked_as_selected_recording ) );

            newRowViewSubviews = new SelectRecordingsListAdapter.ViewHolder();
            newRowViewSubviews.setSelectableRecordingNameView(
                selectableRecordingNameView
            );
            newRowViewSubviews.setMarkedAsSelectedRecordingView(
                markAsSelectedRecordingView
            );

            newRowView.setTag( newRowViewSubviews );
        } else {
            newRowView = convertView;

            newRowViewSubviews =
                ( SelectRecordingsListAdapter.ViewHolder )( convertView.getTag() );
        }

        newRowViewSubviews.
                getSelectableRecordingNameView().
                setText(
                    selectableRecordings.get( position ).first
                );
        newRowViewSubviews.
                getMarkedAsSelectedRecordingView().
                setOnCheckedChangeListener(
                    new SelectedWatcher(
                        selectableRecordings.get( position ).second
                    )
                );

        return( newRowView );
    }

    @Override
    public void notifyDataSetChanged(){
        super.notifyDataSetChanged();

        this.getSelectableRecordings();
    }

    @Override
    public void notifyRecordingsDatabaseChanged() {
        this.notifyDataSetChanged();
    }

    public ArrayList< String > getSelectedRecordings(){
        ArrayList< String > selectedRecordings = new ArrayList< String >();

        for( Pair< String, AtomicBoolean > currentRecordingInfoPair : selectableRecordings){
            if( currentRecordingInfoPair.second.get() == true ){
                selectedRecordings.add(
                    currentRecordingInfoPair.first
                );
            }
        }

        return( selectedRecordings );
    }
}
