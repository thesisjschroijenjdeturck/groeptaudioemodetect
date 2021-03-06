package groept.be.emodetect;

import android.content.Context;
import android.os.Bundle;
import android.preference.ListPreference;
import android.preference.Preference;
import android.preference.PreferenceFragment;
import android.preference.PreferenceManager;

import java.util.HashSet;

import groept.be.emodetect.settingchangelisteners.EmotionClassificationMethodSettingChangeListener;
import groept.be.emodetect.settingchangelisteners.WebServiceURLSettingChangeListener;

public class SettingsActivity extends AppCompatPreferenceActivity {
    public static String WEB_SERVICE_URL_KEY = "web_service_URL";
    public static String DEFAULT_WEB_SERVICE_URL = "http://142.93.133.60/predict";

    public static HashSet< WebServiceURLSettingChangeListener > webServiceURLSettingChangeListeners =
        new HashSet< WebServiceURLSettingChangeListener >();

    public static String WEB_SERVICE_URL_CURRENT_VALUE = "";

    public static class SettingsFragment extends PreferenceFragment {
        @Override
        public void onCreate( Bundle savedInstanceState ){
            super.onCreate( savedInstanceState );

            addPreferencesFromResource( R.xml.settings );

            SettingsActivity.bindWebServiceURLPreference( findPreference( "web_service_URL" ) );
        }
    }

    private static Preference.OnPreferenceChangeListener webServiceURLWatcher = new Preference.OnPreferenceChangeListener(){
        @Override
        public boolean onPreferenceChange( Preference preference, Object newValue ){
            String newValueAsString = newValue.toString();

            SettingsActivity.WEB_SERVICE_URL_CURRENT_VALUE = newValueAsString;

            updateWebServiceURLSettingChangeListeners( newValueAsString );

            preference.setSummary( newValueAsString );

            return( true );
        }
    };

    public static void bindWebServiceURLPreference( Preference preference ){
        preference.setOnPreferenceChangeListener( webServiceURLWatcher );

        Context preferenceContext = preference.getContext();
        String preferenceKey = preference.getKey();
        String preferenceValue =
            PreferenceManager.
                getDefaultSharedPreferences( preferenceContext ).
                getString(
                    preferenceKey,
                    preferenceContext.getString( R.string.default_web_service_URL )
                );

        webServiceURLWatcher.onPreferenceChange( preference, preferenceValue );
    }

    public static void registerWebServiceURLSettingChangeListener( WebServiceURLSettingChangeListener newListener ){
        webServiceURLSettingChangeListeners.add( newListener );

        newListener.updateWebServiceURL( SettingsActivity.WEB_SERVICE_URL_CURRENT_VALUE );
    }

    private static void updateWebServiceURLSettingChangeListeners( String newWebServiceURL ){
        for( WebServiceURLSettingChangeListener currentWebServiceURLChangeListener : webServiceURLSettingChangeListeners ){
            currentWebServiceURLChangeListener.updateWebServiceURL( newWebServiceURL );
        }
    }

    public static String getWebServiceURLFromPreferences( Context applicationContext ){
        String webServiceURL =
            PreferenceManager.
            getDefaultSharedPreferences( applicationContext ).
            getString( WEB_SERVICE_URL_KEY, DEFAULT_WEB_SERVICE_URL );

        return( webServiceURL );
    }

    @Override
    protected void onCreate( Bundle savedInstanceState ){
        super.onCreate( savedInstanceState );

        getFragmentManager().
            beginTransaction().
            replace(
                android.R.id.content,
                new SettingsActivity.SettingsFragment() ).
            commit();
    }
}
