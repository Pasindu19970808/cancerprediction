data = csvread('dataR2.csv');

data = data([2:end],:);

X = data(:,[1:end - 1]);
Y = data(:,end);

%using PCA to visualize the important data
mu = mean(X);
Xadjust = X - mu;

%calculates the covariance matrix for Xadjust
covX = cov(Xadjust);
[eigvec,eigval] = eig(covX);

%calculates the trace which is a representation of the sum of variance of each 
%feature that is used in predicting the cancer 
eigtrace = sum(eigval);
totaltrace = sum(eigtrace);

%as we need more than 99% variance between representative data 
%to visualize the data for important for visualizing the patients
%with cancer, this piece of code calculates the variables 
%that gives 99% variance or above
for i = 1:size(eigtrace,2);
  tracepercent = (sum(eigtrace(:,[1:i])))./totaltrace;
  if i >= 2 && tracepercent > 0.99
    k = i;
    break
  end
end

%Using k and eigvec to get the eigen vectors
eigvec = eigvec(:,[1:2]);
%now we have a 9 x 2 matrix which are the 2 principal components
%capturing 99% or above of our data

%calculating the reduced data
%X is a 116 x 9 matrix
%multiply 116 x 9 matric with the 9 x 2 matrix to get the reduced 
%dimensional matrix
reducedx = Xadjust*eigvec;

%Visualizing the data
%patients without cancer
idxwo = find(Y == 1);
%patients with cancer 
idxwi = find(Y == 2);

figure(1)
%without cancer
plot(reducedx(idxwo,1),reducedx(idxwo,2),'bx');
hold on
%with cancer
plot(reducedx(idxwi,1),reducedx(idxwi,2),'rx');
xlabel('Principal component 1');
ylabel('Principal component 2');

%The major representative variables are found and output as k
Xvisual = X(:,[1:k]);

figure(2)
%without cancer
plot(Xvisual(idxwo,1),Xvisual(idxwo,2),'bx');
hold on
%with cancer
plot(Xvisual(idxwi,1),Xvisual(idxwi,2),'rx');


%output of the plot shows an obvious decision boundary
%provided by these representative variables

%training a logistic regression algorithm to predict cancer
X = data(:,[1:end - 1]);
sd = std(X);
%featurescaling X
Xadjust = Xadjust./sd;

initial_theta = zeros(size(X,2) + 1,1);

Y(idxwo) = 0;
Y(idxwi) = 1;

%As the dataset is too structured with no randomness
%we randomly reorder the dataset
dat = load('randorder.mat');
randorder = dat.A;
Xadjust = Xadjust(randorder,:);
Y = Y(randorder,:);

%A 60%,20%,20% split is used to separate the Training
%set, Validation set and the test set
Xtrain = Xadjust([1:69],:);
Ytrain = Y([1:69],:);

Xval = Xadjust([70:93],:);
Yval = Y([70:93],:);

Xtest = Xadjust([94:end],:);
Ytest = Y([94:end],:);

%regularization parameter
lambda = 0.1;
%Training using the training set
[Jreg,gradreg] = costlogreg(initial_theta,Xtrain,Ytrain,lambda);

%using cost minimization to find the optimal theta values
options = optimset('Gradobj','on','MaxIter',50);

[theta] = fmincg(@(t)(costlogreg(t,Xtrain,Ytrain,lambda)), initial_theta,options);

%calculating accuracy using the Test set
[h] = sigmoid2(Xtest,theta);
list = zeros(size(h,1),1);
idx = find(h >= 0.5);
list(idx) = 1;
accuracy = mean(double(list == Ytest));


%drawing learning curve

listJ = zeros(70,3);
  
for i = 1:70
 
  %collects training set
  initial_theta = zeros(size(Xadjust,2) + 1,1);
  Xtrain = Xadjust([1:i],:);
  Ytrain = Y([1:i],:);
  
  %finds short version for cost function
  costfunction = @(t)costlogreg(t,Xtrain,Ytrain,lambda);
  
  %obtains theta depending on training set
  options = optimset('Gradobj','on','MaxIter',50);
  [theta] = fmincg(costfunction, initial_theta,options);
  
  %calculates the cost on the training set and adds it to a list
  [Jtrain,~] = linearRegCost(theta,Xtrain,Ytrain);
  listJ(i,1) = i;
  listJ(i,2) = Jtrain;
  
  %obtains Jcv using earlier found theta
  [Jval,~] = linearRegCost(theta,Xval,Yval);
  listJ(i,3) = Jval;
end

  figure(3)
  plot(listJ(:,1),listJ(:,2),'b-')
  hold on
  plot(listJ(:,1),listJ(:,3),'r-')
  legend('Training cost','Validation cost');