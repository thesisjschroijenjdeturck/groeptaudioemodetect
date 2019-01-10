package groept.be.emodetect.serviceclients;

public interface WebServiceResultHandler {
    public void handleProperResult( String returnData );
    public void handleException( Exception exception );
}
