function [h] = sigmoid2(X,theta)
  X = [ones(size(X,1),1),X];
  
  z = X*theta;
  
  h = 1./(1 + exp(-1.*z));
  
endfunction
