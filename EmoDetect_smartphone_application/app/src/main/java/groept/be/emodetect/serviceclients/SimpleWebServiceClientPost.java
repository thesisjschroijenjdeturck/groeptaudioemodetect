package groept.be.emodetect.serviceclients;

import android.os.AsyncTask;
import android.util.Log;

import java.net.URL;

public class SimpleWebServiceClientPost extends AsyncTask< String, Integer, String > {
    public final static String WEB_SERVICE_CLIENT_POST_TAG = "WebServiceClientPost";

    private SimpleWebServiceClient webServiceClient;

    private boolean success;

    private WebServiceResultHandler webServiceResultHandler;

    public SimpleWebServiceClientPost( URL webServiceURL, WebServiceResultHandler webServiceResultHandler ){
        super();

        this.webServiceClient = new SimpleWebServiceClient( webServiceURL );
        this.webServiceResultHandler = webServiceResultHandler;

        this.success = false;
    }

    @Override
    protected void onPreExecute() {
        super.onPreExecute();
    }

    @Override
    protected String doInBackground( String... params ) {
        String requestBody = params[ 0 ];

        try {
            success = false;
            Log.d( "ABC", "Calling web service ( POST )... ");
            String webServiceClientResult = webServiceClient.POST( requestBody );
            Log.d( "ABC", "Web service result = " + webServiceClientResult );
            success = true;
            return( webServiceClientResult );
        } catch( Exception e ){
            success = false;
            webServiceResultHandler.handleException( e );
            return( null );
        }
    }

    @Override
    protected void onPostExecute( String result ){
        if( ( success == true ) &&
                ( result != null ) ){
            webServiceResultHandler.handleProperResult( result );
        }
    }
}