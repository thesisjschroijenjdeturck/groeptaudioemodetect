function CCC_joris = CCC_joris(pred,ref)

CCC_joris = [0,0,0,0];
preds = zeros(length(pred),4);

ref_mean=nanmean(ref);
% ref_var=nanvar(ref);
% 
training_std = 1;  %%training data has been normalized so the std is exactly 1 by definition

preds(:,1) = pred;
preds(:,2) = pred-mean(pred)+ref_mean;
preds(:,3) = pred/(training_std*nanstd(pred));
preds(:,4) = (pred-mean(pred)+ref_mean)/(training_std*nanstd(pred));

for i = 1 : 4
 CCC_joris(i) = CCC_calc(preds(:,i),ref); 
end