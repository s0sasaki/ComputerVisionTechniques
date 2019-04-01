clear all;

n = 1000;
alpha1 = 10;
ss1 = 2;
[x1, y1] = generate_points(n, alpha1, ss1);
w = [0];
plot_problem4_b(x1, alpha1, ss1, w);
saveas(gcf, 'problem4b_1.png'); clf;
[x_pca1, w] = pca(x1, n);
plot_problem4_c(x1, x_pca1, alpha1, ss1, w);
saveas(gcf, 'problem4c_1.png'); clf;
[x_lda1, w] = lda(x1, y1, n);
plot_problem4_d(x1, x_lda1, alpha1, ss1, w);
saveas(gcf, 'problem4d_1.png'); clf;

alpha2 = 2;
ss2 = 10;
[x2, y2] = generate_points(n, alpha2, ss2);
w = [0];
plot_problem4_b(x2, alpha2, ss2, w);
saveas(gcf, 'problem4b_2.png'); clf;
[x_pca2, w] = pca(x2, n);
plot_problem4_c(x2, x_pca2, alpha2, ss2, w);
saveas(gcf, 'problem4c_2.png'); clf;
[x_lda2, w] = lda(x2, y2, n);
plot_problem4_d(x2, x_lda2, alpha2, ss2, w);
saveas(gcf, 'problem4d_2.png'); clf;

function [x_lda, w] = lda(x, y, n)
    xx1 = x(y<0.5,:);
    xx2 = x(y>=0.5, :);
    m1 = mean(xx1);
    m2 = mean(xx2);
    s1 = cov(xx1);
    s2 = cov(xx2);
    w = inv(s1+s2)*(m1-m2)';
    w = w/norm(w);
    R = [w(1) -w(2); w(2) w(1)];
    xT = x';
    zT = R' * xT;
    zT(2,:) = zeros(1,n);
    x_ldaT = R * zT;
    x_lda = x_ldaT';
end

function [x_pca, w] = pca(x, n)
    mu_hat = mean(x);
    sigma_hat = cov(x);
    [U,S,V] = svd(sigma_hat);
    x = x';
    z1 = U' * x;
    z1(2,:) = zeros(1, n);
    x_pca = U * z1;
    x = x';
    x_pca = x_pca';
    w = U(:,1);
end

function plot_problem4_d(x, x_lda, alpha, ss, w)
    plot_problem4_c(x, x_lda, alpha, ss, w)
end

function plot_problem4_c(x, x_pca, alpha, ss, w)
    plot_problem4_b(x, alpha, ss, w);
    plot_problem4_b(x_pca, alpha, ss, w);
end

function plot_problem4_b(x, alpha, ss, w)
    hold on;
    title(strcat('alpha=', int2str(alpha), ' sigma^2=', int2str(ss)));
    scatter(x(:,1), x(:,2));
    if (w(1) ~= 0)
        plot([-100*w(1);100*w(1)], [-100*w(2);100*w(2)]);
    end
    xlim([-15 15]);
    ylim([-15 15]);
    hold off;    
end

function [x, y] = generate_points(n, alpha, ss)
    mu1 = [alpha; 0];
    mu2 = -mu1;
    sigma = [1 0; 0 ss];
    x1 = mvnrnd(mu1, sigma, n);
    x2 = mvnrnd(mu2, sigma, n);
    x = zeros(n,2);
    y = rand(1, n);
    for i=1:n
        if(y(i)<0.5)
            x(i,:) = x1(i,:);
        else
            x(i,:) = x2(i,:);
        end
    end
end
