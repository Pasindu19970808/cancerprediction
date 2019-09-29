function [] = plotregboundary(theta,X,degree);
  %[features] = createfeatures(X, degree);
  X1 = X(:,1);
  X2 = X(:,2);
  
  X1 = linspace(-1,1.5,50);
  X2 = linspace(-1,1.5,50);
  
  [x1,x2] = meshgrid(X1,X2);
  hvals = zeros(size(x1,1),size(x2,1));
  k = 0;
  for i = 1:size(x1,1)
    k = k + 1;
    for j = 1:size(x2,1)
      x = [x1(1,i), x2(j,i)];
      [features] = createfeatures(x,degree);
      z = features*theta;
  
      h = 1./(1 + exp(-1.*z));
      
      hvals(j,k) = h;
    end
  end

  figure(1)
  contour(x1,x2,hvals,[0.5 1])
  figure(2)
  surf(x1,x2,hvals)
  
endfunction
