package groept.be.emodetect.serviceclients.dtos;

public class PredictResult {
    private String recordingID;
    private float arousal;
    private float valence;

    public PredictResult( String recordingID, float arousal, float valence ){
        this.recordingID = recordingID;
        this.arousal = arousal;
        this.valence = valence;
    }

    public String getRecordingID(){
        return( this.recordingID );
    }

    public float getArousal(){
        return( this.arousal );
    }

    public float getValence(){
        return( this.valence );
    }
}
