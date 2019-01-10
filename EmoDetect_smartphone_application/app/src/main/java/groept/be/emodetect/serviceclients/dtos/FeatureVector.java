package groept.be.emodetect.serviceclients.dtos;

public class FeatureVector {
    private int frameOffset;
    private float[] features;

    public FeatureVector( int frameOffset, float[] features){
        this.frameOffset = frameOffset;
        this.features = features;
    }
}
