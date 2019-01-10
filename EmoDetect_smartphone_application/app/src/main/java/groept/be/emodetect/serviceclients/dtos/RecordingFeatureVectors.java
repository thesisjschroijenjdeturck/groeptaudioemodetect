package groept.be.emodetect.serviceclients.dtos;

import java.util.ArrayList;
import java.util.List;

public class RecordingFeatureVectors {
    private String recordingID;
    private List< FeatureVector > mfccVectors;

    public RecordingFeatureVectors( String recordingID, ArrayList< FeatureVector > mfccVectors ){
        this.recordingID = recordingID;
        this.mfccVectors = mfccVectors;
    }
}
