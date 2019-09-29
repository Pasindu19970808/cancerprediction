function [features] = createfeatures(X, degree);
  X1 = X(:,1);
  X2 = X(:,2);
  
  features = ones(size(X1(:,1)));
  
  for i = 1:degree
    for j = 0:i
      features(:,end+1) = (X1.^(i - j)).*(X2.^j);
    end
  end
  
  
endfunction
