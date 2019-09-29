function [J,grad] = costlog(theta,X,Y)
  
  X = [ones(size(X,1),1),X];
  grad = zeros(size(X,2),1);
  
  m = size(Y,1);
  
  z = X*theta;
  
  h = 1./(1 + exp(-1.*z));
  
  J = (1/m)*((-1*(Y'*(log(h)))) - ((1-Y)'*((log(1 - h)))));
  
  grad = (1/m).*(X'*(h - Y));
  
endfunction
