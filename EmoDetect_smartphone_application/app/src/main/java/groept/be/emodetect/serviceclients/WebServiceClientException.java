package groept.be.emodetect.serviceclients;

import java.io.FileNotFoundException;

public class WebServiceClientException extends Exception {
    private static final long serialVersionUID = 1L;

    public WebServiceClientException( String message ) {
        super( message );
    }

    public WebServiceClientException( String message, Throwable cause ){
        super( message, cause );
    }

    @Override
    public String getMessage(){
        String errorMessage;

        if( getCause() instanceof FileNotFoundException ){
            // We received a 404 error
            errorMessage = "No endpoint for web service URL defined by server!";
        } else {
            errorMessage = super.getMessage();
        }

        return( errorMessage );
    }
}
