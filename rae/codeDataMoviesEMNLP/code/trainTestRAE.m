%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for
% Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions
% Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning
% Conference on Empirical Methods in Natural Language Processing (EMNLP 2011)
% See http://www.socher.org for more information or to ask questions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load minFunc
addpath(genpath('tools/'))

%%%%%%%%%%%%%%%%%%%%%%
% Hyperparameters
%%%%%%%%%%%%%%%%%%%%%%
% set this to 1 to train the model and to 0 for just testing the RAE features (and directly training the classifier)
params.trainModel = 1;

% node and word size
params.embedding_size = 50;

% Relative weighting of reconstruction error and categorization error
params.alpha_cat = 0.2;

% Regularization: lambda = [lambdaW, lambdaL, lambdaCat, lambdaLRAE];
params.lambda = [1e-05, 0.0001, 1e-07, 0.01];

% weight of classifier cost on nonterminals
params.beta=0.5;

func = @norm1tanh;
func_prime = @norm1tanh_prime;

% parameters for the optimizer
options.Method = 'lbfgs';
options.display = 'on';
options.maxIter = 70;

disp(params);
disp(options);




%%%%%%%%%%%%%%%%%%%%%%
% Pre-process dataset
%%%%%%%%%%%%%%%%%%%%%%
% set this to different folds (1-10) and average to reproduce the results in the paper
params.CVNUM = 1;
preProFile = ['../data/rt-polaritydata/RTData_CV' num2str(params.CVNUM) '.mat'];

% read in polarity dataset
if ~exist(preProFile,'file')
    read_rtPolarity
else
    load(preProFile, 'labels','train_ind','test_ind', 'cv_ind','We2','allSNum','test_nums');
end
sent_freq = ones(length(allSNum),1);
[~,dictionary_length] = size(We2);

% split this current fold into train and test
index_list_train = cell2mat(allSNum(train_ind)');
index_list_test = cell2mat(allSNum(test_ind)');
index_list_cv = cell2mat(allSNum(cv_ind)');
unq_train = sort(index_list_train);
unq_cv = sort(index_list_cv);
unq_test = sort(index_list_test);
freq_train = histc(index_list_train,1:size(We2,2));
freq_cv = histc(index_list_cv,1:size(We2,2));
freq_test = histc(index_list_test,1:size(We2,2));
freq_train = freq_train/sum(freq_train);
freq_cv = freq_cv/sum(freq_cv);
freq_test = freq_test/sum(freq_test);

cat_size=1;% for multinomial distributions this would be >1
numExamples = length(allSNum(train_ind));



%%%%%%%%%%%%%%%%%%%%%%
% Initialize parameters
%%%%%%%%%%%%%%%%%%%%%%
theta = initializeParameters(params.embedding_size, params.embedding_size, cat_size, dictionary_length);


%%%%%%%%%%%%%%%%%%%%%%
% Parallelize if on cluster
%%%%%%%%%%%%%%%%%%%%%%
if isunix && matlabpool('size') == 0
    numCores = feature('numCores');
    if numCores==16
        numCores=8;
    end
    matlabpool('open', numCores);
end


%%%%%%%%%%%%%%%%%%%%%%
% Train/Test Model
%%%%%%%%%%%%%%%%%%%%%%
if params.trainModel
    lbl = labels(:,train_ind);
    snum = allSNum(train_ind);
    sent_freq_here = sent_freq(1:numExamples);
    
    [opttheta, cost] = minFunc( @(p)RAECost(p, params.alpha_cat, cat_size,params.beta, dictionary_length, params.embedding_size, ...
        params.lambda, We2, snum, lbl, freq_train, sent_freq, func, func_prime), ...
        theta, options);
    theta = opttheta;
    
    [W1, W2, W3, W4, b1, b2, b3, Wcat,bcat, We] = getW(1, theta, params.embedding_size, cat_size, dictionary_length);
    
    save(['../output/savedParams_CV' num2str(params.CVNUM) '.mat'],'opttheta','params','options');
    classifyWithRAE
    
else
    if params.CVNUM ~= 1
        error('This is the optimal file for CV-fold 1')
    end
    load('../output/optParams_RT_CV1.mat')
    classifyWithRAE
end
