function [x] = OMPnorm(A,y,K)
% Orthogonal Matching Pursuit (OMP)
% Solve y = Ax, assuming x is a sparse vector with sparsity numOMP.
% [x] = OMPnorm(A,y,numOMP)
% [INPUTS]
% A - dictionary (N x M)
% y - Data vector of size M x 1
% K - desired sparsity (number of atoms)
% [OUTPUTS]
% x - Sparse coefficient matrix of size M x 1

[N, M] = size(A);

%Anorm = %normalize dictionary A such that all columns have norm equal to 1
Anorm = A ./ vecnorm(A);

x = zeros(M,1);
indx = zeros(K,1);

residual = y;                           % initial residual is full measurement vector
for j = 1:K
    %proj     = %calculate dot product between residual and each column of Anorm 
    proj     = residual'*Anorm;
    %[~,pos]  = %find index number of maximum value of proj
    [~,pos]  = max(proj);
    pos      = pos(1);                  % choose first value if multiple correlations are equal
    indx(j)  = pos;                     % store indices
    
    %a        = %least squares estimate of x using only columns of A given by indx
    a        = pinv(A(:, indx(1:j)))*y;
    %residual = %residual = signal minus projection
    residual = y - A(:, indx(1:j))*a;
end

x(indx(1:j)) = a;
