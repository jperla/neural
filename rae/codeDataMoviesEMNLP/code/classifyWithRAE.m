
theta = opttheta;

[W1, W2, W3, W4, b1, b2, b3, Wcat,bcat, We] = getW(1, theta, params.embedding_size, cat_size, dictionary_length);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Get features by forward propagating and finding structure for all train and test sentences...')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in this setting, we take the top node's vector and the average of all vectors in the tree as a concatenated feature vector

fulltraining_instances = getFeatures(allSNum(train_ind),0,...
    We,We2,W1,W2,W3,W4,b1,b2,b3,Wcat,bcat,params.alpha_cat,params.embedding_size, ...
    labels(:,train_ind), freq_train, func, func_prime, params.trainModel);

fulltesting_instances = getFeatures(allSNum(test_ind),0,...
    We,We2,W1,W2,W3,W4,b1,b2,b3,Wcat,bcat,params.alpha_cat,params.embedding_size,...
    labels(:,test_ind), freq_test, func, func_prime, params.trainModel);

training_labels = labels(:,train_ind)';
testing_labels = labels(:,test_ind)';


[t1 t2 t3] = size(fulltraining_instances);
training_instances = reshape(fulltraining_instances,t1, t2*t3);
[t1 t2 t3] = size(fulltesting_instances);
testing_instances = reshape(fulltesting_instances,t1,t2*t3);

[num_training_instances ~] = size(training_instances);
[num_testing_instances ~] = size(testing_instances);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Logistic regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize parameters
theta2 = -0.5+rand(t3*params.embedding_size*cat_size + cat_size,1);

options2.Method = 'lbfgs';
options2.maxIter = 1000;

% Training
[theta2, cost] = minFunc( @(p) soft_cost(p,training_instances, training_labels, 1e-6),theta2, options2);
b = theta2(end);
W = theta2(1:end-1)';

dec_val = sigmoid(W*training_instances' + b(:,ones(num_training_instances,1)));
pred = 1*(dec_val > 0.5);
gold = training_labels';
[prec_train, recall_train, acc_train, f1_train] = getAccuracy(pred, gold);

% Testing
dec_val = sigmoid(W*testing_instances' + b(:,ones(num_testing_instances,1)));
pred = 1*(dec_val > 0.5);
gold = testing_labels';
[prec_test, recall_test, acc_test, f1_test] = getAccuracy(pred, gold);

acc_train
acc_test

fid = fopen('../output/resultsRAE.txt','a');
fprintf(fid,[num2str(params.CVNUM),',',num2str(1),',',num2str(params.embedding_size),',',num2str(params.lambda(1)),',',num2str(params.lambda(2)), ',' ...,
    num2str(params.lambda(3)),',',num2str(params.alpha_cat),',',...
    num2str(options.maxIter)]);

fprintf(fid,',train,%f,%f,%f,%f',acc_train, prec_train, recall_train, f1_train);
fprintf(fid,',test,%f,%f,%f,%f\n',acc_test, prec_test, recall_test, f1_test);
fclose(fid);
