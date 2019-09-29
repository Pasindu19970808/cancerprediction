data = load('ex2data1.txt');

X1 = data(:,1);
X2 = data(:,2);
Y = data(:,3);

figure(1)
hold on
plot2(X1,X2,Y)

X = data(:,[1:end - 1]);

initial_theta = zeros(size(X,2) + 1,1);

%test_theta = [-24; 0.2; 0.2];
[h] = sigmoid2(X1,X2,initial_theta);

[J,grad] = costlog(initial_theta,X,Y);

options = optimset('Gradobj','on','MaxIter',400);

[theta,cost] = fminunc(@(t)(costlog(t,X,Y)), initial_theta,options);

[x1,x2,hvals] = plotboundary(theta,X);

figure(1)
contour(x1, x2, hvals,[0.5 1])

[list] = admission(theta,X);

accuracy = mean(double(list == Y));

m = size(X,1);
listJtrain = zeros(m - 1,2);
listJcv = zeros(m - 1, 2);
%drawing learning curve
for i = 1:m
 
  %collects training set
  initial_theta = zeros(size(X,2) + 1,1);
  Xtrain = X([1:i],:);
  Ytrain = Y([1:i],:);
  
  %collects validation set
  Xval = X([80:end],:);
  
  
  %finds short version for cost function
  costfunction = @(t)costlog(t,Xtrain,Ytrain);
  
  options = optimset('Gradobj','on','MaxIter',400);
  %obtains theta depending on training set
  [theta] = fmincg(costfunction, initial_theta,options);
  
  listJtrain(i,1) = i;
  listJtrain(i,2) = Jtrain;
  %collects cv set
  temp = i + 1;
  Xcv = X([temp:end],:);
  Ycv = Y([temp:end],:);
  %obtains Jcv using earlier found theta
  [Jcv,grad] = costlog(theta,Xcv,Ycv);
  listJcv(i,1) = i;
  listJcv(i,2) = Jcv;
  figure(2)
  plot(i,Jtrain,'bx')
  hold on
  plot(i,Jcv,'rx')
  
end
