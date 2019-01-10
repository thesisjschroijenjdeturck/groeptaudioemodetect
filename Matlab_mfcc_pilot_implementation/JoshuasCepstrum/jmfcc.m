function mfccs = jmfcc( signal, sampleRate )

%{
We need to load Octaves signal package because
it contains the dct() function for computing
the DCT of the log filterbank energies!
%}
pkg load signal;

frameLength = 0.025; % We use a DSP standard 25 ms frame length
frameStep   = 0.010; % We use a DSP standard 10 ms frame step

signalFrames = ...
  jframesignal( signal, sampleRate, frameLength, frameStep );
numberOfFrames = size( signalFrames, 1 );

FFTLength = 512;
%FFTLength = 2 ^ ceil( log2( frameLength * sampleRate ) );
signalFFTs = zeros( numberOfFrames, FFTLength ); 
signalPeriodograms = zeros( numberOfFrames, FFTLength );
for currentFrame = 1 : numberOfFrames
  currentFFTnumbers = ...
    fft( signalFrames( currentFrame, : ), FFTLength );
  signalFFTs( currentFrame, : ) = currentFFTnumbers;
  signalPeriodograms( currentFrame, : ) = ( abs( currentFFTnumbers ) .^ 2 ) / FFTLength;
end
periodogramLength = ...
  ( ( FFTLength / 2 ) + 1 );
signalPeriodograms( :, ( periodogramLength + 1 ) : FFTLength ) = [];

melFilterBank = transpose( jmelbank( 26, 300, 8000, sampleRate, periodogramLength ) );

logFilterBankEnergies = ...
  zeros( numberOfFrames, size( melFilterBank, 1 ) );
for currentMelFilterIndex = 1 : size( melFilterBank, 1 )

  for currentPeriodogramIndex = 1 : size( signalPeriodograms, 1 )
    logFilterBankEnergies( currentPeriodogramIndex, currentMelFilterIndex ) = ...
      log10( ...
        sum( ...
          signalPeriodograms( currentPeriodogramIndex, : ) .* ...
          melFilterBank( currentMelFilterIndex, : ) ...
        ) ...
      );
  end

end

%{
The transposing and untransposing is necessary
because our log filterbank energies are stored in
rows while the dct() function interprets columns as
signals!
%}
logFilterBankEnergies = transpose( dct( transpose( logFilterBankEnergies ) ) );

%{
For MFCC's, we are only interested in the first 12 DCT coefficients
%}
logFilterBankEnergies( :, 13 : size( logFilterBankEnergies, 2 ) ) = [];

mfccs = logFilterBankEnergies;

end