function hz = jmel2hz( frequencyInMel )
  hz = ( 700 * ( e .^ ( frequencyInMel / 1125 ) - 1 ) );
end