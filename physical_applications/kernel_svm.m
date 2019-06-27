function [a, b] = softsvm_RBF(X, t, C)
  
  N = size(X,1);
  K = (t*t');
  for i=1:N
      for j=1:N
        K(i,j) = K(i,j)*exp(-norm(X(i,:)-X(j,:))^2);
      end
  end
  a = quadprog(2*K, -1*ones(N,1), zeros(1,N),0, t',0, zeros(N,1),C*ones(N,1));
  b = (t - X * X' * (a .* t))/N;

end