clear all;

X = readtrainset();
U = pca(X);
pred_pca(X,U);
W = rda(X, 1.0, true);
pred_rda(X, W);
pred_pca_lda(X,U);

function pred_pca_lda(X,U)
    k = 30;
    muX = mean(X);
    Z = U(:,1:k)' * (X-muX)';
    Z = Z';
    W = lda(Z);
    muZ = mean(Z);
    A = W * (Z-muZ)';
    A = A';
    
    [m,s] = calc_param(A);
    Xt = readtestset();
    Zt = U(:,1:k)' * (Xt-muX)';
    Zt = Zt';
    At = W * (Zt-muZ)';
    At = At';
    
    pred = gaussian_classification(At, m, s);
    disp_accuracy(pred);
end
    
function pred_rda(X, W)
    mu = mean(X);
    Z = W * (X-mu)';
    Z = Z';
    [m,s] = calc_param(Z);
    Xt = readtestset();
    Zt = W * (Xt-mu)';
    Zt = Zt';
    pred = gaussian_classification(Zt, m, s);
    disp_accuracy(pred);
end

function pred_pca(X, U)
    mu = mean(X);
    Z = U(:,1:15)' * (X-mu)';
    Z = Z';
    [m,s] = calc_param(Z);
    Xt = readtestset();
    Zt = U(:,1:15)' * (Xt-mu)';
    Zt = Zt';
    pred = gaussian_classification(Zt, m, s);
    disp_accuracy(pred);
end

function [m,s] = calc_param(Z)
    m = zeros(6, 15);
    s = zeros(6, 15, 15);
    for i = 0:5
        Z1 = Z(40*i+1:40*i+40, :);
        m1 = mean(Z1);
        s1 = cov(Z1-m1);
        m(i+1, :) = m1;
        s(i+1, :,:) = s1;
    end
end

function pred = gaussian_classification(Zt, m, s)
    pred = zeros(1, 6*10);
    for i = 1:6*10
        z = Zt(i,:);
        pmax = -Inf;
        for j = 1:6
            mj = m(j,:);
            sj = s(j,:,:);
            sj = reshape(sj, [15 15]);            
            p = -(z-mj)*inv(sj)*(z-mj)'-log(det(sj));
            if(p>pmax)
                pmax = p;
                pred(i) = j;
            end
        end
    end
end

function disp_accuracy(pred)
    c = 0;
    for i=0:5
        for j=1:10
            if (pred(10*i+j) == i+1)
                c = c+1;
            end
        end
    end
    disp(c/60);
end

function W = lda(X)
    W = rda(X, 0.0, false);
end

function W = rda(X, gamma, print_flag)
    n = size(X,2);
    W = zeros(15, n);
    p=0;
    for i = 0:5
        for j = i+1:5
            X1 = X(40*i+1:40*i+40, :);
            X2 = X(40*j+1:40*j+40, :);
            m1 = mean(X1);
            m2 = mean(X2);
            s1 = cov(X1-m1);
            s2 = cov(X2-m2);
            w = inv(s1+s2+gamma*eye(n))*(m2-m1)';
            w = w/norm(w);
            p = p+1;
            W(p, :) = w;
            if(print_flag)
                subplot(4,4,p);
                imshow(reshape(w, [50 50])*2550);
            end
        end
    end
    if(print_flag)
        print(gcf, '-djpeg', 'problem5b.jpg');
        clf;
    end
end

function U = pca(X)
    mu = mean(X);
    sigma = cov(X-mu);
    [U,S,V] = svd(sigma);
    for i=1:16
        subplot(4,4,i);
        img = reshape(U(:,i), [50, 50]);
        imshow(img*255);
    end
    print(gcf, '-djpeg', 'problem5a.jpg');
    clf;
end

function X = readtestset()
    X = zeros([6*10 2500]);
    for i=0:5
        for j=1:10
            filename = strcat('testset\subset',int2str(i+6),'\person_',int2str(i+1),'_',int2str(j),'.jpg');
            A = imread(filename);
            A = reshape(A, [1 50*50]);
            X(10*i+j, :) = A;
        end
    end
end

function X = readtrainset()
    X = zeros([6*40 2500]);
    for i=0:5
        for j=1:40
            filename = strcat('trainset\subset',int2str(i),'\person_',int2str(i+1),'_',int2str(j),'.jpg');
            A = imread(filename);
            A = reshape(A, [1 50*50]);
            X(40*i+j, :) = A;
        end
    end
end

