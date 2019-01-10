package groept.be.emodetect.helpers.analysishelpers;

import groept.be.emodetect.helpers.miscellaneous.ArrayTools;
import be.tarsos.dsp.io.android.AudioDispatcherFactory;
import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.mfcc.MFCC;
import groept.be.emodetect.helpers.databasehelpers.RecordingsDatabaseHelper;
import groept.be.emodetect.helpers.databasehelpers.featuresdatabases.MfccsDatabaseHelper;

import android.content.Context;
import android.support.v4.util.Pair;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.io.IOException;

// MfccExtractor( 44100, 1024, 128, 40, 50, 300, 3000 );
public class MfccExtractor implements FeatureExtractor {
    final int sampleRate;
    final int bufferSize;
    final int bufferOverlap;
    final int noDctCoefficients;
    final int noMelFilters;
    final float lowerFrequency;
    final float upperFrequency;

    final int bufferStep;

    final RecordingsDatabaseHelper recordingsDatabaseHelper;
    final MfccsDatabaseHelper mfccsDatabaseHelper;

    public MfccExtractor(
        Context applicationContext,
        int sampleRate,
        int bufferSize,
        int bufferOverlap,
        int noDctCoefficients,
        int noMelFilters,
        float lowerFrequency,
        float upperFrequency ){
        this.sampleRate = sampleRate;
        this.bufferSize = bufferSize;
        this.bufferOverlap = bufferOverlap;
        this.noDctCoefficients = noDctCoefficients;
        this.noMelFilters = noMelFilters;
        this.lowerFrequency = lowerFrequency;
        this.upperFrequency = upperFrequency;

        this.bufferStep = ( bufferSize - bufferOverlap );

        this.recordingsDatabaseHelper = RecordingsDatabaseHelper.getInstance( applicationContext );
        this.mfccsDatabaseHelper = MfccsDatabaseHelper.getInstance( applicationContext );
    }

    @Override
    public ArrayList< Pair< Integer, float[] > > extractFeatures( final String recordingAbsoluteFilename ) throws IOException {
        final ArrayList< Pair< Integer, float[] > > mfccResults = new ArrayList< Pair< Integer, float[] > >();

        AudioDispatcher dispatcher = AudioDispatcherFactory.fromPipe( recordingAbsoluteFilename, sampleRate, bufferSize, bufferOverlap );

        final MFCC mfcc =
            new MFCC(
                this.bufferSize,        // Samples per frame
                this.sampleRate,        // Sample rate
                this.noDctCoefficients, // Amount of DCT coefficient
                this.noMelFilters,      // Amount of mel filters
                this.lowerFrequency,    // Amount of lower frequency
                this.upperFrequency     // Amount of upper frequency
            );

        dispatcher.addAudioProcessor( mfcc );
        dispatcher.addAudioProcessor( new AudioProcessor(){
            int currentFrameStartingSample = 0;

            @Override
            public void processingFinished() {
                String recordingRelativeFilename = ( new File( recordingAbsoluteFilename ) ).getName();

                // Calculate delta coefficients
                int N = 2;

                for( int currentResultIndex = 0; currentResultIndex < mfccResults.size(); ++currentResultIndex ){
                    float[] currentResultArray = mfccResults.get( currentResultIndex ).second;
                    float[] deltaCoefficients = new float[ MfccExtractor.this.noDctCoefficients ];

                    int deltaCoefficientIndex = 0;
                    for( int coefficientIndex = 0; coefficientIndex < MfccExtractor.this.noDctCoefficients; ++coefficientIndex ){
                         deltaCoefficientIndex = coefficientIndex;
                         float numerator = 0;
                         float denominator = MfccExtractor.this.getDeltaDenominator( N );
                         for( int n = 1; n <= N; ++n ){
                             float forwardCoefficient = MfccExtractor.this.getCoefficientInOtherFrame( mfccResults, currentResultIndex, n, coefficientIndex );
                             float backwardCoefficient = MfccExtractor.this.getCoefficientInOtherFrame( mfccResults, currentResultIndex, -n, coefficientIndex );

                             numerator += ( ( forwardCoefficient - backwardCoefficient ) * n );
                         }

                         deltaCoefficients[ deltaCoefficientIndex ] = numerator / denominator;
                    }

                    Pair< Integer, float[] > updatedCurrentResult =
                        new Pair< Integer, float[] >(
                            mfccResults.get( currentResultIndex ).first,
                            ArrayTools.mergeFloatArrays( currentResultArray, deltaCoefficients )
                        );
                    mfccResults.set( currentResultIndex, updatedCurrentResult );
                }

                // Calculate delta-delta coefficients (N already defined previously)
                for( int currentResultIndex = 0; currentResultIndex < mfccResults.size(); ++currentResultIndex ){
                    float[] currentResultArray = mfccResults.get( currentResultIndex ).second;
                    float[] deltaDeltaCoefficients = new float[ MfccExtractor.this.noDctCoefficients ];

                    int deltaDeltaCoefficientIndex = 0;
                    for( int coefficientIndex = MfccExtractor.this.noDctCoefficients; coefficientIndex < ( 2 * MfccExtractor.this.noDctCoefficients ); ++coefficientIndex ){
                        deltaDeltaCoefficientIndex = ( coefficientIndex - MfccExtractor.this.noDctCoefficients );
                        float numerator = 0;
                        float denominator = MfccExtractor.this.getDeltaDenominator( N );
                        for( int n = 1; n <= N; ++n ){
                            float forwardCoefficient = MfccExtractor.this.getCoefficientInOtherFrame( mfccResults, currentResultIndex, n, coefficientIndex );
                            float backwardCoefficient = MfccExtractor.this.getCoefficientInOtherFrame( mfccResults, currentResultIndex, -n, coefficientIndex );

                            numerator += ( ( forwardCoefficient - backwardCoefficient ) * n );
                        }

                        deltaDeltaCoefficients[ deltaDeltaCoefficientIndex ] = numerator / denominator;
                    }

                    Pair< Integer, float[] > updatedCurrentResult =
                            new Pair< Integer, float[] >(
                                    mfccResults.get( currentResultIndex ).first,
                                    ArrayTools.mergeFloatArrays( currentResultArray, deltaDeltaCoefficients )
                            );
                    mfccResults.set( currentResultIndex, updatedCurrentResult );
                }

                Log.d("ABC",mfccResults.size() + " features extracted");
                mfccsDatabaseHelper.insertFeatures(
                        recordingsDatabaseHelper.getRecordingID( recordingRelativeFilename ),
                        mfccResults
                );
            }

            @Override
            public boolean process( AudioEvent audioEvent ){
                mfccResults.add( new Pair< Integer, float[] >( currentFrameStartingSample, mfcc.getMFCC() ) );
                currentFrameStartingSample += bufferStep;
                return( true );
            }
          }
        );

        dispatcher.run();

        return( mfccResults );
    }

    private float getCoefficientInOtherFrame( ArrayList< Pair< Integer, float[] > > resultPairs, int frameIndex, int frameOffset, int coefficientIndex ) {
        int resultPairsLength = resultPairs.size();

        if (((frameIndex + frameOffset) >= 0 &&
                (frameIndex + frameOffset) < resultPairsLength)) {
            int indexOfInterest = (frameIndex + frameOffset);

            Pair<Integer, float[]> resultPairOfInterest = resultPairs.get(indexOfInterest);
            return resultPairOfInterest.second[coefficientIndex];
        } else {
            return ((float) (0));
        }
    }

    private int getDeltaDenominator( int N ) {
        int deltaDenominator = 0;
        for( int n = 1; n <= N; ++n ){
            deltaDenominator += ( n * n );
        }
        deltaDenominator *= 2;

        return deltaDenominator;
    }
}
