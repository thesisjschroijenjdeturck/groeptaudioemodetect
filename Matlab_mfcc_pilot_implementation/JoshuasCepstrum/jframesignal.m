%{
A simple function for dividing a signal into (overlapping) frames

Written by Joshua Schroijen
%}
function frames = jframesignal( signal, sampleRate, frameLength, frameStep ) 

samplesPerFrame = floor( sampleRate * frameLength );
samplesStep = floor( sampleRate * frameStep );

disp( samplesPerFrame );
disp( samplesStep );

workingFrames = [];
signalIndex = 1;

while signalIndex <= ( length( signal ) - samplesPerFrame + 1 )
  workingFrames = ...
    [ workingFrames ; signal( signalIndex : ( signalIndex + samplesPerFrame - 1 ) ) ];
  signalIndex = signalIndex + samplesStep;

end

frames = workingFrames;

end