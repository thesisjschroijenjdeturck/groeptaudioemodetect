function smooth_joris = smooth_joris(input_data,half_window_length)

smooth_joris = zeros(length(input_data),1);

for i = 1 : length(input_data)
   if(  (i > half_window_length) && (i < length(input_data) - half_window_length)  ) 
        smooth_joris(i,1) = mean( input_data(i- half_window_length: i+ half_window_length,1) );
   else
       if i <= half_window_length        
        smooth_joris(i,1) = mean( input_data(1:i+half_window_length,1) );
       else
        smooth_joris(i,1) =  mean( input_data(i-half_window_length:length(input_data),1) );
       end
   end
    
end