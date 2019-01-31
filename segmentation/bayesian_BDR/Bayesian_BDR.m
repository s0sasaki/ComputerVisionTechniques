clear all;

global cheetah;
global cheetah_mask;
global zz;
global alpha;

load('Alpha.mat');
load('TrainingSamplesDCT_subsets_8.mat');
prior1 = load('Prior_1.mat');
prior2 = load('Prior_2.mat');
cheetah = im2double(imread('cheetah.bmp'));
cheetah_mask = im2double(imread('cheetah_mask.bmp'));
zz = load('Zig-Zag Pattern.txt');


res1 = execute_all_alpha(D1_FG, D1_BG, prior1, 'D1, strategy1');
res2 = execute_all_alpha(D2_FG, D2_BG, prior1, 'D2, strategy1');
res3 = execute_all_alpha(D3_FG, D3_BG, prior1, 'D3, strategy1');
res4 = execute_all_alpha(D4_FG, D4_BG, prior1, 'D4, strategy1');
result_prior1 = vertcat(res1, res2, res3, res4);
csvwrite('result_prior1.csv', result_prior1);
res1 = execute_all_alpha(D1_FG, D1_BG, prior2, 'D1, strategy2');
res2 = execute_all_alpha(D2_FG, D2_BG, prior2, 'D2, strategy2');
res3 = execute_all_alpha(D3_FG, D3_BG, prior2, 'D3, strategy2');
res4 = execute_all_alpha(D4_FG, D4_BG, prior2, 'D4, strategy2');
result_prior2 = vertcat(res1, res2, res3, res4);
csvwrite('result_prior2.csv', result_prior2);

function result = execute_all_alpha(D_FG, D_BG, prior, description)
    global alpha;

    err_bayes_D_prior_arr = [];
    err_ml_D_prior_arr    = [];
    err_map_D_prior_arr   = [];
    for a = alpha
        disp(a);
        [err_bayes_D_prior, err_ml_D_prior, err_map_D_prior] = train_and_test(a, D_FG, D_BG, prior, description);
        err_bayes_D_prior_arr = [err_bayes_D_prior_arr, err_bayes_D_prior];
        err_ml_D_prior_arr    = [err_ml_D_prior_arr, err_ml_D_prior];
        err_map_D_prior_arr   = [err_map_D_prior_arr, err_map_D_prior];
    end
    
    figure;
    semilogx(alpha, err_bayes_D_prior_arr, alpha, err_ml_D_prior_arr, alpha, err_map_D_prior_arr);
    legend('Bayes', 'ML', 'MAP');
    saveas(gcf, strcat(description, '.png'));
    clf;
    
    result = vertcat(err_bayes_D_prior_arr, err_ml_D_prior_arr, err_map_D_prior_arr);
end


function [error_rate_bayes, error_rate_ml, error_rate_map] = train_and_test(a, D_FG, D_BG, prior, description)

    nsample_fg = size(D_FG, 1);
    nsample_bg = size(D_BG, 1);
    p_prior_fg = nsample_fg/(nsample_bg + nsample_fg);
    p_prior_bg = nsample_bg/(nsample_bg + nsample_fg);

    sigma_fg = calc_sigma(D_FG);
    sigma_bg = calc_sigma(D_BG);
    [mu1_fg, sigma1_fg] = calc_param1(D_FG, prior.mu0_FG, prior.W0, a, sigma_fg);
    [mu1_bg, sigma1_bg] = calc_param1(D_BG, prior.mu0_BG, prior.W0, a, sigma_bg);
    
    % Question(a) % Bayes
    A = makemask(p_prior_fg, p_prior_bg, mu1_fg, mu1_bg, sigma_fg+sigma1_fg, sigma_bg+sigma1_bg);
    error_rate_bayes = eval_mask(A, p_prior_fg, p_prior_bg, strcat(description, ', Bayes'));
    
    % Question(b) % ML
    %[mu_old_fg, sigma_old_fg] = calc_param_old(D_FG);
    %[mu_old_bg, sigma_old_bg] = calc_param_old(D_BG);
    %A = makemask(p_prior_fg, p_prior_bg, mu_old_fg, mu_old_bg, sigma_old_fg, sigma_old_bg);
    mu2_fg = mean(D_FG);
    mu2_bg = mean(D_BG);
    A = makemask(p_prior_fg, p_prior_bg, mu2_fg, mu2_bg, sigma_fg, sigma_bg);
    error_rate_ml = eval_mask(A, p_prior_fg, p_prior_bg, strcat(description, ', ML'));    
    
    % Question(c) % MAP
    A = makemask(p_prior_fg, p_prior_bg, mu1_fg, mu1_bg, sigma_fg, sigma_bg);
    error_rate_map = eval_mask(A, p_prior_fg, p_prior_bg, strcat(description, ', MAP'));

end


function error_rate = eval_mask(A, p_prior_fg, p_prior_bg, description)
    global cheetah_mask;
    
    imagesc(A);
    colormap(gray(255));
    title(description);
    clf(figure);

    disp(description);
    error = xor(A, cheetah_mask);
    error_rate = sum(sum(error))/(255*270);
    disp(error_rate);

    error_fg = and(error, cheetah_mask);
    error_bg = and(error, not(cheetah_mask));
    error_rate_fg = sum(sum(error_fg))/sum(sum(cheetah_mask));
    error_rate_bg = sum(sum(error_bg))/sum(sum(not(cheetah_mask)));
    error_rate2 = p_prior_fg*error_rate_fg + p_prior_bg*error_rate_bg;
%     disp(error_rate_fg);
%     disp(error_rate_bg);
%     disp(error_rate2);
end


function A = makemask(p_prior_fg, p_prior_bg, mu1_fg, mu1_bg, sigma_fg, sigma_bg)
    global cheetah;
    global zz;
    
    A = zeros(size(cheetah));
    for i = 1:size(cheetah, 1)
        for j = 1:size(cheetah, 2)
            if 5<i && i<size(cheetah, 1)-4 && 5<j && j<size(cheetah, 2)-4
                block = cheetah(i-4:i+3, j-4:j+3);
                block_dct = abs(dct2(block, 8, 8));

                xdata = zeros(1, 64);
                for u = 1:8
                    for v = 1:8
                        xdata(zz(u, v)+1) = block_dct(u,v);
                    end
                end

                p_cond_fg = mvnpdf(xdata, mu1_fg, sigma_fg);
                p_cond_bg = mvnpdf(xdata, mu1_bg, sigma_bg);

                if p_cond_fg * p_prior_fg > p_cond_bg * p_prior_bg
                    A(i, j) = 1;
                end
            end
        end
    end
end


function [mu, sigma] = calc_param_old(data)
    sigma = cov(data);
    mu = mean(data);
end

function [mu1, sigma1] = calc_param1(data, mu0, w0, a, sigma)
    sigma0 = diag(a * w0);
    n = size(data, 1);
    %sigma1_inv = n * inv(sigma) + inv(sigma0);
    %sigma1 = inv(sigma1_inv);
    
    mu1T = sigma0 * inv(sigma0 + sigma./n) * mean(data)' + (1/n) * sigma * inv(sigma0 + sigma/n) * mu0'; 
    sigma1 = sigma0 * inv(sigma0 + sigma./n) * sigma./n;
    mu1 = mu1T';

    %mu1 = (-1)* (sum(data)*inv(sigma) + mu0 * inv(sigma0)) * sigma1;
    %mu1T = sigma1 * (inv(sigma)*sum(data)' + inv(sigma0)*mu0');
    %mu1 = mu1T';
end

function sigma = calc_sigma(data)
%     n = size(data, 1);
%     m = mean(data);
%     data0 = data-m;
%     sigma = zeros(64,64);
%     for i = 1:n
%         sigma = sigma + data0(i,:)' * data0(i,:) ;
%     end
%     sigma = sigma/n;
    sigma = cov(data, 1);
end
