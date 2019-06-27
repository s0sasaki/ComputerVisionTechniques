%% Non-Uniform Sampling

clear;
clc;
close all;

% random seed
rng(1);

%Signal frequencies (Hz)
f1 = 1e3;
f2 = 2e3;
f3 = 4e3;

fs = 10e3;               %Sampling frequency
T  = 1/fs;               %Time between samples
N  = 1000;               %Number of samples
t  = (0:N-1)'.*T;        %PTime of each sample

%Generate time domain signal corrupted with noise
s_original = 1*sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.7*sin(2*pi*f3*t);
s = s_original + 0.2*randn(N,1);

%Choose M random samples of the signal
M = 101;                    %number of random measurements
k = randperm(N);            %indices of random samples
m = sort(k(1:M));           %save just the first M indices in sorted order
y = s(m);                   %sample signal at random points

%Plot time-domain signal
figure(1);
subplot(3,2,1);
plot(t,s,'r-',t(m),y,'bo');
xlabel('Time (s)');
ylabel('s(t)');
grid on;
axis([0,.02,-2.5,2.5]);
title('Original s(t) + Noise');

%Plot (sparse) Fourier domain signal
X = fft(s_original);

subplot(3,2,2);
plot(linspace(-fs/2,fs/2,length(X)),abs(fftshift(X)));
xlabel('Frequency (Hz)');
ylabel('|X(f)|');
grid on;
axis([-fs/2,fs/2,0,520]);
title('Original X(f)');

%Generate linear transform matrix
PHI = fft(eye(N));                          %DFT matrix (OR use dftmtx(N))
E = zeros(M,N);                             
E(sub2ind(size(E),1:M,m)) = 1;              %S has entries of 1 where sampled
A = E*(1/N)*PHI';                           %Appropriately scrambled DFT matrix

%% Naive l2 minimization solution
Xl2 = pinv(A)*y;
sl2 = (1/N)*real(PHI'*Xl2);      %Reconstruct signal

%Plot frequency domain naive (L2) solution
subplot(3,2,4);
plot(linspace(-fs/2,fs/2,length(X)),abs(fftshift(Xl2)));
xlabel('Frequency (Hz)');
ylabel('|X_{L_2}(f)|');
grid on;
axis([-fs/2,fs/2,0,520]);
title('L2 reconstruction (f)');

%Plot time domain naive (L2) solution
subplot(3,2,3);
plot(t,sl2,'r-');
xlabel('Time (s)');
ylabel('s_{L_2}(t)');
grid on;
axis([0,.02,-2.5,2.5]);
title('L2 reconstruction (t)');

%% Approximate sparse solution using OMP with sparsity between 1-20.
%  Choose solution with lowest reconstruction error
bestnumOMP = 0;
max_numOMP = 20;
error = zeros(1,max_numOMP);
error_min = norm(s);
for numOMP = 1:max_numOMP
    X_omp = OMPnorm(A,y,numOMP);
    s_omp = (1/N)*real(PHI'*X_omp);
    
    error(numOMP) = norm(s-s_omp);
    
    if error(numOMP) < error_min
        bestnumOMP = numOMP;
        error_min = error(numOMP);
    end
end

disp(strcat('Best sparsity for OMP = ',num2str(bestnumOMP)));

% Plot the best solution using OMP
X_omp_best = OMPnorm(A,y,bestnumOMP);
s_omp_best = (1/N)*real(PHI'*X_omp_best);      %Reconstruct signal

%Plot frequency domain Matching Pursuit solution
subplot(3,2,6);
plot(linspace(-fs/2,fs/2,length(X)),abs(fftshift(X_omp_best)));
xlabel('Frequency (Hz)');
ylabel('|X_{OMP}(f)|');
grid on;
axis([-fs/2,fs/2,0,520]);
title('OMP Reconstruction (f)');

%Plot time domain Matching Pursuit solution
subplot(3,2,5);
plot(t,s_omp_best,'r-');
xlabel('Time (s)');
ylabel('s_{OMP}(t)');
title('OMP Reconstruction (t)');
grid on;
axis([0,.02,-2.5,2.5]);

%% plot OMP reconstruction error vs sparsity
figure;
plot(1:max_numOMP,error,'ko-');
xlabel('Sparsity');
ylabel('Reconstruction error');
title('Reconstruction error vs. Sparsity (OMP)')
grid on;

%end