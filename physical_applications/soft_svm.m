function [a, b] = softsvm(X, t, C)
  
  K = (X*X') .* (t*t');
  N = size(K,1);
  a = quadprog(2*K, -1*ones(N,1), zeros(1,N),0, t',0, zeros(N,1),C*ones(N,1));
  b = (t - X * X' * (a .* t))/N;

end