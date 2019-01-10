function mel = jhz2mel( frequencyInHz )
  mel = ( 1125 * log( 1 + ( frequencyInHz / 700 ) ) );
end