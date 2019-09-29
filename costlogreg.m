function [Jreg,gradreg] = costlogreg(theta,X,Y,lambda)
  X = [ones(size(X,1),1),X];
  
  gradreg = zeros(size(X,2),1);
  
  m = size(Y,1);
  
  z = X*theta;
  
  h = 1./(1 + exp(-1.*z));
  
  J1 = (1/m).*((-1*(Y'*(log(h)))) - ((1-Y)'*((log(1 - h)))));
  J2 = (lambda./(2*m))*(sum(theta([2:end],:).^2));
  Jreg = J1 + J2;
  
  gradreg(1,:) = (1/m).*(X(:,1)'*(h - Y));
  
  gradreg([2:end],:) = (1/m).*(X(:,[2:end])'*(h - Y)) ...
       + (lambda/m).*theta([2:end],:);
endfunction
