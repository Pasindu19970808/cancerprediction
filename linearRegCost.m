function [J] = linearRegCost(theta,X,Y)

m = length(Y);

X = [ones(size(X,1),1),X];
  
z = X*theta;

h = 1./(1 + exp(-1.*z));

J = (1/(2*m))*sum((h - Y).^2);

end