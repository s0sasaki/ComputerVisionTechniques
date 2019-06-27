%% Generate data
clear all
clc
rng(0)
load 'K.mat'

N           = 100;
D           = 2;
num_S1      = 50;
num_S2      = N-num_S1;

S1          = randn(num_S1,D);
tmp         = 4*exp(1i*2*pi*(rand(num_S2,1)));
S2          = [real(tmp)+rand(num_S1,1), imag(tmp)+rand(num_S1,1)];
X           = [S1;S2];
t           = ones(N,1);
t(1:num_S1) = -1;



%% Classify

[alpha, bias] = softsvm_RBF(X, t, .01);

z           = sign(K*(alpha.*t));
C1_est      = find(z==1);
C2_est      = find(z==-1);


figure(1)
clf
subplot(2,1,1)
scatter(S1(:,1),S1(:,2),'r')
hold on
scatter(S2(:,1),S2(:,2),'b')
title('True classes')
xlabel('x')
ylabel('y')
grid on
subplot(2,1,2)
scatter(X(C2_est,1),X(C2_est,2),'r')
hold on
scatter(X(C1_est,1),X(C1_est,2),'b')
title('Your Radial Basis SVM classifications')
xlabel('x')
ylabel('y')
grid on
