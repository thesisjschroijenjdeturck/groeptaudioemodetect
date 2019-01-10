%% set path correctly

addpath(genpath('C:/Users/Joris/Documents/MATLAB'))     % Add this folder and its subfolders to working directory

%% READ IN MFCC's of training data **********************************************************
%*******************************************************************************************

data_path = 'C:\Users\Joris\Documents\MATLAB\RECOLA\' ; % path that points to the RECOLA folder
addpath(genpath(data_path));                            % put database in working directory 

x_train_arousal = zeros(7501, 89);
x_train_valence = zeros(7501, 89);
x_train_arousal_tmp = zeros(7501, 89);
x_train_valence_tmp = zeros(7501, 89);


x_dev_arousal = zeros(7501, 89);                        % all files have a length of 7501
x_dev_valence = zeros(7501, 89);
x_dev_arousal_tmp = zeros(7501, 89);
x_dev_valence_tmp = zeros(7501, 89);


y_train_arousal = zeros(7501, 1); 
y_train_valence = zeros(7501, 1); 
y_train_arousal_tmp = zeros(7501, 89);
y_train_valence_tmp = zeros(7501, 89);


y_dev_arousal = zeros(7501, 1);        
y_dev_valence = zeros(7501, 1);
y_dev_arousal_tmp = zeros(7501, 89);
y_dev_valence_tmp = zeros(7501, 89);

% last 75 frames are discarded for the features, and first 75 labels to
% compensate for annotation delay for each file  (75 frames = 3s)
for i=1:9
    if(i==1)
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\arousal\train_', i, '.arff');
        x_train_arousal = read_my_arff(arff_path, 92, 7592);
        x_train_arousal = x_train_arousal(1:7426, :);  % 40 ms/frame, 3 sec delay = 75 frames
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\valence\train_', i, '.arff');
        x_train_valence = read_my_arff(arff_path, 92, 7592);
        x_train_valence = x_train_valence(1:7426, :);
        
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\arousal\dev_', i, '.arff');
        x_dev_arousal = read_my_arff(arff_path, 92, 7592);
        x_dev_arousal = x_dev_arousal(1:7426, :);
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\valence\dev_', i, '.arff');
        x_dev_valence = read_my_arff(arff_path, 92, 7592);
        x_dev_valence = x_dev_valence(1:7426, :);
        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\arousal\train_', i, '.arff');
        y_train_arousal = read_my_arff_labels(arff_path, 6, 7506);
        y_train_arousal = y_train_arousal(76:end, :);        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\valence\train_', i, '.arff');
        y_train_valence = read_my_arff_labels(arff_path, 6, 7506);
        y_train_valence = y_train_valence(76:end, :);
        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\arousal\dev_', i, '.arff');
        y_dev_arousal = read_my_arff_labels(arff_path, 6, 7506);
        y_dev_arousal = y_dev_arousal(76:end, :);
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\valence\dev_', i, '.arff');
        y_dev_valence = read_my_arff_labels(arff_path, 6, 7506);
        y_dev_valence = y_dev_valence(76:end, :);
    else
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\arousal\train_', i, '.arff');
        x_train_arousal_tmp = read_my_arff(arff_path, 92, 7592);
        x_train_arousal = [x_train_arousal; x_train_arousal_tmp(1:7426, :)];
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\valence\train_', i, '.arff');
        x_train_valence_tmp = read_my_arff(arff_path, 92, 7592);
        x_train_valence = [x_train_valence; x_train_valence_tmp(1:7426, :)];
        
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\arousal\dev_', i, '.arff');
        x_dev_arousal_tmp = read_my_arff(arff_path, 92, 7592);
        x_dev_arousal = [x_dev_arousal; x_dev_arousal_tmp(1:7426, :)];
        arff_path = sprintf('%s%s%d%s', data_path,'features_audio\valence\dev_', i, '.arff');
        x_dev_valence_tmp = read_my_arff(arff_path, 92, 7592);
        x_dev_valence = [x_dev_valence; x_dev_valence_tmp(1:7426, :)];
        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\arousal\train_', i, '.arff');
        y_train_arousal_tmp = read_my_arff_labels(arff_path, 6, 7506);    
        y_train_arousal = [y_train_arousal; y_train_arousal_tmp(76:end, :)];
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\valence\train_', i, '.arff');
        y_train_valence_tmp = read_my_arff_labels(arff_path, 6, 7506);    
        y_train_valence = [y_train_valence; y_train_valence_tmp(76:end, :)];
        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\arousal\dev_', i, '.arff');
        y_dev_arousal_tmp = read_my_arff_labels(arff_path, 6, 7506);    
        y_dev_arousal = [y_dev_arousal; y_dev_arousal_tmp(76:end, :)];
        
        arff_path = sprintf('%s%s%d%s', data_path,'ratings_gold_standard\valence\dev_', i, '.arff');
        y_dev_valence_tmp = read_my_arff_labels(arff_path, 6, 7506);    
        y_dev_valence = [y_dev_valence; y_dev_valence_tmp(76:end, :)];
    end
end
% clear unneeded variables

clearvars x_train_arousal_tmp x_train_valence_tmp x_dev_arousal_tmp x_dev_valence_tmp ...
    y_train_arousal_tmp y_train_valence_tmp y_dev_arousal_tmp y_dev_valence_tmp arff_path i;

%

% DELAY COMPENSATION  (ALREADY IMPLEMENTED ABOVE) 
% 4 first seconds of labels where dropped and last 4 seconds of features were dropped! 
%******************************************************************************************
% FROM LITERATURE ON SEWA: "The optimal delay compensation, estimated using multimodal
% features, was found to be 4s for arousal and 2s for valence."
%
% "... temporal shifts were applied for each file in the training set in order to realign the features
% with the ground truth. The frame shift was achieved by dropping first N ground truth scores and last 
% N input feature frames.."
%******************************************************************************************

% RANDOMIZATION AND NORMALIZATION
% From here on all pre-processing is done, delay compensation is performed and
% the sizes of X and Ytrain are matched 

% also randomize the sequence of the training data and the labels
sel = randperm(size(x_train_arousal, 1));
x_train_arousal = x_train_arousal(sel,:) ;
x_train_valence = x_train_valence(sel,:) ;

y_train_arousal = y_train_arousal(sel,:) ;
y_train_valence = y_train_valence(sel,:) ;

clearvars sel; 

% normalize the features with respect to training mean and std
stdX_arousal = std(x_train_arousal);
stdX_valence = std(x_train_valence);       
meanX_arousal = mean(x_train_arousal);
meanX_valence = mean(x_train_valence);

stdY_arousal = std(y_train_arousal);
stdY_valence = std(y_train_valence);       
meanY_arousal = mean(y_train_arousal);
meanY_valence = mean(y_train_valence);

for index = 1 : 88
  x_train_arousal(:, index) = (x_train_arousal(:, index)-meanX_arousal(1, index))/stdX_arousal(1, index);
  x_train_valence(:, index) = (x_train_valence(:, index)-meanX_valence(1, index))/stdX_valence(1, index);
  x_dev_arousal(:, index) = (x_dev_arousal(:, index)-meanX_arousal(1, index))/stdX_arousal(1, index);
  x_dev_valence(:, index) = (x_dev_valence(:, index)-meanX_valence(1, index))/stdX_valence(1, index);
end

% normalize arousal labels
y_train_arousal = (y_train_arousal - meanY_arousal)/stdY_arousal;
y_train_valence = (y_train_valence - meanY_valence)/stdY_valence;
y_dev_arousal = (y_dev_arousal - meanY_arousal)/stdY_arousal;
y_dev_valence = (y_dev_valence - meanY_valence)/stdY_valence;

clearvars index stdX_arousal stdX_valence stdY_arousal stdY_valence meanX_arousal meanX_valence ...
meanY_arousal meanY_valence;

%% Fixed size LS-SVM part starts here
% The fixed size LS-SVM is based on two ideas (see also Section 2.4): the first is to exploit the
% primal-dual formulations of the LS-SVM in view of a Nystr¨om approximation 

% The second one is to do active support vector selection (here based on entropy criteria). The

%% Determine the best size of the subset

% Optimal values for the kernel parameters and the capacity of the fixed size LS-SVM can be
% obtained using a simple Monte Carlo experiment. For different kernel parameters and capacities
% (number of chosen support vectors), the performance on random subsets of support vectors are
% evaluated. The means of the performances are minimized by an exhaustive search (Figure 3.15b):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save_path = 'C:\Users\Joris\Documents\MATLAB\perf_recola\' ;  % path to save grid search results too

% set target to either arousal or valence

target = 'valence';  % must be a string !!        

if(target == 'arousal') 
    x_train = x_train_arousal;
    x_dev = x_dev_arousal;
    y_train = y_train_arousal;
    y_dev = y_dev_arousal;    
else
      if ( target == 'valence')
        x_train = x_train_valence;
        x_dev = x_dev_valence;
        y_train = y_train_valence;
        y_dev = y_dev_valence;
      else
        printf("invalid target was specified");
      end
      
end


caps = [50 100 200];                               %  first a gridsearch on size of subset and 
sig2s = [100 200 500 1000 3000 5000 8000];         %  sigma² is performed
nb = 4;

performances_CCC_develop = zeros(length(caps),length(sig2s));
performances_CCC_develop_scaled = zeros(length(caps),length(sig2s));
performances_develop = zeros(nb,1);
performances_develop_scaled = zeros(nb,1);

performances_CCC_train = zeros(length(caps),length(sig2s));
performances_CCC_train_scaled = zeros(length(caps),length(sig2s));
performances_train = zeros(nb,1);
performances_train_scaled = zeros(nb,1);

used_gams = zeros(length(caps),length(sig2s));
tmp_gams = zeros(nb,1);

for i=1:length(caps)
    for j=1:length(sig2s)
     for t = 1:nb
       fprintf('iteration outer-loop: %d/%d, middle loop: %d/%d, inner loop: %d/%d \n',i,length(caps),j,length(sig2s),t,nb);
       sel = randperm(size(x_train, 1));                     % create vector with randomized indices
       svX = x_train(sel(1:caps(i)),:);                      % select random permutation of inputs & labels
       svY = y_train(sel(1:caps(i)),:);                      % get corresponding labels as well
       features = AFEm(svX,'RBF_kernel',sig2s(j), x_train);  % map training data to feature space 
       [Cl3, gam_opt] = bay_rr(features, y_train, 1, 3);     % find optimal gamma using bayesian inference
       tmp_gams(t) = gam_opt;                                % keep track of the optimal gamma's
       [W,b] = ridgeregress(features, y_train, gam_opt);     % perform a ridge regression in feature space
       features_svX = AFEm(svX,'RBF_kernel',sig2s(j), svX);          % map svX and Devel-data to feature space as well
       features = AFEm(svX,'RBF_kernel',sig2s(j), x_dev);
       Yh = features*W+b;                                            % get Devel prediction
       Yh_train = features_svX*W+b;                                  % get Train/svX prediction
       Yh_scaled = (Yh - mean(Yh))/nanstd(Yh) ;                      % rescaled prediction   *********
       Yh_scaled_train = (Yh_train - mean(Yh_train))/nanstd(Yh_train);
       performances_develop(t) =  CCC_calc(Yh, y_dev);               % performance without scaling
       performances_train(t) =  CCC_calc(Yh_train, svY);                            
       performances_develop_scaled(t) =  CCC_calc(Yh_scaled, y_dev); % performance with scaling
       performances_train_scaled(t) =  CCC_calc(Yh_scaled_train, svY);              
     end
     used_gams(i, j) = mean(tmp_gams);                                % save gams
     performances_CCC_develop(i, j) = mean(performances_develop);     % save CCC's for each combination of caps and sig2s
     performances_CCC_train(i, j) = mean(performances_train);         % average CCC over nb random permutations is used
     performances_CCC_develop_scaled(i, j) = mean(performances_develop_scaled);
     performances_CCC_train_scaled(i, j) = mean(performances_train_scaled);
   end
end

fprintf('finished grid search on capacity and sigma \n')
save = strcat(save_path, 'caps.csv');     %save caps
csvwrite(save, caps);                     
save = strcat(save_path, 'sig2s.csv');    %save sig2s
csvwrite(save, sig2s);
save = strcat(save_path, 'develop_ccc.csv');  % save CCCs for train and develop
csvwrite(save, performances_CCC_develop);     % scaled and smoothed 
save = strcat(save_path, 'train_ccc.csv');
csvwrite(save, performances_CCC_train);
save = strcat(save_path, 'develop_ccc_scaled.csv');
csvwrite(save, performances_CCC_develop_scaled);
save = strcat(save_path, 'train_ccc_scaled.csv');
csvwrite(save_path, performances_CCC_train_scaled);
msg = sprintf('%s%s', 'saved results in this folder -> ', save_path);
printf(msg)
%% some plotting ................
% plot CCC's in function of sigma
figure; 
plot(sig2s(1:7), performances_CCC_train_scaled(1, 1:7), sig2s(1:7), performances_CCC_develop(1, 1:7));
title('Training and Development CCC as a function of sigma²');
xlabel('sigma²') 
ylabel('Concordance Correlation Coëfficient') 
legend('Training set', 'Development set', 'Location', 'SouthEast');

%% train model once more with best performing hyper-paramaters
sigmaatje = 1000;        % best peforming sig2
capsje = 100;            % best subset size
sel = randperm(size(x_train, 1));                     % create vector with randomized indices
svX = x_train(sel(1:capsje),:);                      % select random permutation of inputs & labels
svY = y_train(sel(1:capsje),:);                      % get corresponding labels as well
features = AFEm(svX,'RBF_kernel', sigmaatje, x_train);  % map training data to feature space 
[Cl3, gam_opt] = bay_rr(features, y_train, 1, 3);     % find optimal gamma using bayesian inference
[W,b] = ridgeregress(features, y_train, gam_opt);     % perform a ridge regression in feature space
features = AFEm(svX,'RBF_kernel', sigmaatje, x_dev);
Yh = features*W+b;
%
ceetje = CCC_calc(Yh, y_dev);

%% plot the resulting prediction
str = sprintf('%s%d%s%ds%f', 'Best prediction on Development set, subset size = ', capsje, ', sigma = ', sigmaatje,' and CCC = ', ceetje);
figure;
plot(1:length(y_dev), y_dev, 1:length(y_dev), Yh);
title(str);
xlabel('sample number'); 
ylabel(target); 
legend('Ground Truth', 'Prediction', 'Location', 'SouthEast');


%% select the best subset of X 
% initiate values
Nc=100;                        % this is the number of samples in the subset
subset_indices = 1:Nc;         % create column vector with initial indices 1 -> 200

kernel = 'RBF_kernel';
sigma2 = 7500;
crit_old=-inf;
Xs = x_train(subset_indices,:);
Ys = y_train(subset_indices,:);
disp(' The optimal reduced set is constructed iteratively: ');
% iterate over data
tel2 = 0;
for tel=1:length(x_train)
    
      if rem(tel,100) == 0 || tel2 == 0   % printf to follow progress during execution
        fprintf('iteration #%d/%d just started \n',tel2*100, length(x_train));
        tel2 = tel2+1 ;
      end  
      % new candidate set
      Xsp=Xs; Ysp=Ys;
      % S=ceil(length(x_train)*rand(1));
      Sc=ceil(Nc*rand(1));
      Xs(Sc,:) = x_train(tel,:);
      Ys(Sc) = y_train(tel);  
      
      % automaticly extract features and compute entropy
      crit = kentropy(Xs,kernel, sigma2);
      if crit <= crit_old
        crit = crit_old;
        Xs=Xsp;
        Ys=Ysp;  
      else
        crit_old = crit;
        subset_indices(Sc) = tel;    % update subset_indices is one of the "original" support vectors is replaced 
      end  
end

%% TRY AND DETERMINE THE BEST VALUES FOR sigma and gamma

gammarray = [1]; % dummy either we use bay_rr() to determine gamma or we include it in the grid search  
sigmarray = [7500];
opt_gams =  1:length(sigmarray);

perf_subset_train = zeros(length(gammarray),length(sigmarray));       % matrices to save performance on train & develop
perf_subset_develop = zeros(length(gammarray),length(sigmarray));     % with and without scaling 
perf_subset_train_scaled = zeros(length(gammarray),length(sigmarray)); 
perf_subset_develop_scaled = zeros(length(gammarray),length(sigmarray));

for i = 1 : length(gammarray)                                         % loop over all gamma canidate-values
  for j = 1 : length(sigmarray)                                       % loop over all sigma candidate-values    
       fprintf('outer-loop: %d/%d , inner-loop: %d/%d \n',i ,length(gammarray), j, length(sigmarray));
       features = AFEm(Xs, 'RBF_kernel', sigmarray(j), x_train);       
       [Cl3, gam_opt] = bay_rr(features, y_train,1,3);        % determine optimal gamma with bayesian inference
       opt_gams(j) = gam_opt;                                 % keep track of chosen gamma
       [W,b] = ridgeregress(features, y_train, gam_opt);      % do a ridge regression
       features = AFEm(Xs, 'RBF_kernel', sigmarray(j), Xs);
       Yh_train = features*W+b;                                       % get train prediction
       Yh_train_scaled = (Yh_train - nanmean(Yh_train))/nanstd(Yh_train); % rescaling
       features = AFEm(Xs,'RBF_kernel',sigmarray(j), x_dev);  % map devel data to feature space
       Yh = features*W+b;                                             % get devel prediction
       Yh_scaled = (Yh - nanmean(Yh))/nanstd(Yh);                     % rescaling
       perf_subset_train(i, j) = CCC_calc(Ys, Yh_train);              % save the CCC's for training and development
       perf_subset_train_scaled(i, j) = CCC_calc(Ys, Yh_train_scaled);% also scaled predictions are saved
       perf_subset_develop(i, j) = CCC_calc(y_dev, Yh);                     
       perf_subset_develop_scaled(i, j) = CCC_calc(y_dev, Yh_scaled);    
  end
end
%% plots
% CCC
figure; 
plot(sigmarray, perf_subset_train_scaled(1, :) , sigmarray, perf_subset_develop(1, :));
title('Training and Development CCC as a function of sigma², subset constructed with Renyi entropy criterion');
xlabel('sigma²') 
ylabel('Concordance Correlation Coëfficient') 
legend('Training set', 'Development set', 'Location', 'SouthEast');

%% save performances
strcat(save_path)
csvwrite('C:\Users\Joris\Documents\MATLAB\perf_recola\valence\sigmarray.csv', sigmarray);
csvwrite('C:\Users\Joris\Documents\MATLAB\perf_recola\valence\develop_ccc.csv', perf_subset_develop); % save hyper-params
csvwrite('C:\Users\Joris\Documents\MATLAB\perf_recola\valence\train_ccc.csv', perf_subset_train);
csvwrite('C:\Users\Joris\Documents\MATLAB\perf_recola\valence\develop_ccc_scaled.csv', perf_subset_develop_scaled);
csvwrite('C:\Users\Joris\Documents\MATLAB\perf_recola\valence\train_ccc_scaled.csv', perf_subset_train_scaled);

%%  plotting
figure;
x_as = 1:length(y_dev);
plot(x_as, y_dev, x_as, Yh);
title('Best prediction on Development set with actively selected subset, subset size = 100, sigma = 7500 and CCC = 0.6965');
xlabel('sample number') 
ylabel('Arousal') 
legend('Ground Truth', 'Prediction', 'Location', 'SouthEast');

%% Multi Dimensional Scaling
%  1) take random sample of subset, size = 10k samples ?
sze = 10000;
getter_indx = randperm(length(x_train));
getter_indx = getter_indx(1:10000);
getter_indx = [subset_indices, getter_indx];
Xmds = x_train(getter_indx, :);
%%  2) MDS on random samples + 200 selected SV's
% Create a dissimilarity matrix.
dissimilarities = pdist(Xmds,'euclidean');
%%   convert to square form
dissimilarities = squareform(dissimilarities);
%%
[Xreduced,e] = cmdscale(dissimilarities, 2);
%
%% 3) Plot random samples in blue and SV's in red, random permutation in green
rand_idx = randperm(10100);
figure;
plot(Xreduced(101:10100, 1), Xreduced(101:10100, 2), 'b.', Xreduced(1:100, 1), Xreduced(1:100, 2), 'ro', ...
    Xreduced(rand_idx(1:100), 1), Xreduced(rand_idx(1:100), 2), 'go');
title('Plot of input data after using multidimensional scaling to bring the input dimension down to 2');
legend('Large random sample of input data', 'Actively selected subset', 'Random sample of input data with same size as subset' , 'Location', 'SouthEast');


