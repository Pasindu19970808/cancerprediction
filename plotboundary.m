function [x1,x2,hvals] = plotboundary(theta,X)
  X1 = X(:,1);
  X2 = X(:,2);
  X1range = linspace(min(X1),max(X1),100);
  X2range = linspace(min(X2),max(X2),100);
  
  [x1,x2] = meshgrid(X1range,X2range);
  hvals = zeros(size(x1,1),size(x1,2));
  for i = 1:size(x1,1)
    k = 0;
    for j = 1:size(x2,1)
      k = k + 1;
      xtemp = [1,x1(k,i),x2(j,k)];
      z = xtemp*theta;
      h = 1./(1 + exp(-1.*z));
      hvals(j,i) = h;
    endfor
  endfor
endfunction
