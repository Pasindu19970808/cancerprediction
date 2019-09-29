datareg = load('ex2data2.txt');

X1 = datareg(:,1);
X2 = datareg(:,2);
Y =  datareg(:,3);


pass = find(Y == 1);
fail = find(Y == 0);

figure(1)
plot(X1(pass),X2(pass),'bx');
hold on 
plot(X1(fail),X2(fail),'rx');
legend('pass','fail');

X = [X1,X2];
degree = 6;
[features] = createfeatures(X,degree);


X = features;
initial_theta = zeros(size(features,2),1);
lambda = 1;

[Jreg,gradreg] = costlogreg(initial_theta,X,Y,lambda);


options = optimset('Gradobj','on','MaxIter',400);

[theta] = fmincg(@(t)costlogreg(t,X,Y,lambda),initial_theta,options);

X = [X1,X2];

plotregboundary(theta,X,degree);
