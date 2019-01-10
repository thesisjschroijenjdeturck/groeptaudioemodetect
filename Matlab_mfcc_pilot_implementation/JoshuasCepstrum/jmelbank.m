%{
This function is meant to be used in the context of MFCC computation!

This function will generate a vector containing the points specifying
the frequencies where the different mel filters should have their base
points and top point at.
PLEASE NOTE THAT THE FREQUENCIES PASSED INTO THIS FUNCTION SHOULD NOT
BE CONVERTED TO MELS FIRST!
For the rest, the parameters are self-explanatory

The return value of this function is a matrix where each column is the
FFT of one mel filter. Its dimension is therefore naturally
the FFT length specified by the number of mel filters requested.

Written by Joshua Schroijen
%}

function melbank = jmelbank( numberOfFilters, minimumFrequency, maximumFrequency, sampleRate, FFTLength )

filters = zeros( FFTLength, numberOfFilters );

%{
When determining the frequency points for the triangular mel filters,
we should note that this is done by dividing up the frequency range
of interest equally between the mel filters with the frequencies
converted into mel-frequencies beforehand!
%}

minimumFrequencyMel = jhz2mel( minimumFrequency );
maximumFrequencyMel = jhz2mel( maximumFrequency );
melFrequencyRange = ( maximumFrequencyMel - minimumFrequencyMel );
filterMels = [ minimumFrequencyMel : ( melFrequencyRange / ( numberOfFilters + 2 - 1 ) ) : maximumFrequencyMel ];
filterHzs = jmel2hz( filterMels );

filterBins = floor( ( FFTLength * filterHzs ) / sampleRate );

minimumFrequencyBin = floor( ( FFTLength * minimumFrequency ) / sampleRate );
maximumFrequencyBin = floor( ( FFTLength * maximumFrequency ) / sampleRate );

%{
Here we loop over each filter point. For each one, we generate
a triangle around it, making the filter that is centered on that point
if applicable ( i.e. it is not the first or last point ).
This is why the index starts at 2 and ends at numberOfFilters + 1:
these points correspond to the tops of the first and last mel filter,
respectively. Every point outside of this interval should be ignored.
%}

for currentFilterCenterFrequency = 2 : ( numberOfFilters + 1 )
  currentFilter = ( currentFilterCenterFrequency - 1 );

  for currentBin = minimumFrequencyBin : maximumFrequencyBin

    if currentBin >= filterBins( currentFilterCenterFrequency - 1 ) && ...
       currentBin <= filterBins( currentFilterCenterFrequency )

      filters( currentBin, currentFilter ) = ...
        ( currentBin - filterBins( currentFilterCenterFrequency - 1 ) ) / ...
        ( filterBins( currentFilterCenterFrequency ) - filterBins( currentFilterCenterFrequency - 1 ) );  

    elseif currentBin >= filterBins( currentFilterCenterFrequency ) && ...
           currentBin <= filterBins( currentFilterCenterFrequency + 1 )
      
      filters( currentBin, currentFilter ) = ...
        ( filterBins( currentFilterCenterFrequency + 1 ) - currentBin ) / ...
        ( filterBins( currentFilterCenterFrequency + 1 ) - filterBins( currentFilterCenterFrequency ) );

    end

  end

end

melbank = filters;

end