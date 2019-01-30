clear all;
global ts;
global cheetah;
global cheetah_mask;
global p_prior_fg;
global p_prior_bg;
global mean_fg;
global mean_bg;
global var_fg;
global var_bg;
global zz;

ts = load('TrainingSamplesDCT_8_new.mat');
cheetah = im2double(imread('cheetah.bmp'));
cheetah_mask = im2double(imread('cheetah_mask.bmp'));
zz = load('Zig-Zag Pattern.txt');

nsample_fg = size(ts.TrainsampleDCT_FG, 1);
nsample_bg = size(ts.TrainsampleDCT_BG, 1);
p_prior_fg = nsample_fg/(nsample_bg + nsample_fg);
p_prior_bg = nsample_bg/(nsample_bg + nsample_fg);
display(p_prior_fg);
display(p_prior_bg);

mean_fg = mean(ts.TrainsampleDCT_FG);
mean_bg = mean(ts.TrainsampleDCT_BG);
var_fg = mean(ts.TrainsampleDCT_FG .^ 2) - mean_fg.^2; 
var_bg = mean(ts.TrainsampleDCT_BG .^ 2) - mean_bg.^2; 

hold on; 
for i = 0:7
    for j = 1:8
        k = 8*i+j;
        ax = subplot(8,8,k);
        x = -5:0.01:5;

        m_fg = mean_fg(k);
        v_fg = var_fg(k);
        y_fg = exp(-((x-m_fg).^2)/v_fg)/sqrt(2*pi*v_fg);    
        m_bg = mean_bg(k);
        v_bg = var_bg(k);
        y_bg = exp(-((x-m_bg).^2)/v_bg)/sqrt(2*pi*v_bg);   

        plot(x,y_fg,'--b', x,y_bg, 'g');
        title('feature'+string(k));
        xlim(ax, [-0.1 0.1]);
    end
end
clf(figure);


% best_indices = 1:8;
% best_indices = [1 2 3 4 5 6 26 8];
% best_indices = [1 2 3 6 7 8 9 14];
best_indices = [1 2 3 6 7 8 14 26];
worst_indices = 57:64;
all_indices = 1:64;

display(best_indices);
display(worst_indices);
plot8(best_indices);
plot8(worst_indices);

eval_indices(best_indices);
eval_indices(all_indices);


function plot8(indices)
    global mean_fg;
    global mean_bg;
    global var_fg;
    global var_bg;
    global zz;
    hold on; 
    for i = 1:8
        ax = subplot(4,2,i);
        k = indices(i);
        x = -5:0.01:5;

        m_fg = mean_fg(k);
        v_fg = var_fg(k);
        y_fg = exp(-((x-m_fg).^2)/v_fg)/sqrt(2*pi*v_fg);    
        m_bg = mean_bg(k);
        v_bg = var_bg(k);
        y_bg = exp(-((x-m_bg).^2)/v_bg)/sqrt(2*pi*v_bg);   

        plot(x,y_fg,'--b', x,y_bg, 'g');
        title('feature'+string(k));
        if k == 1
            xlim(ax, [-5 5]);
        else
            xlim(ax, [-0.1 0.1]);
        end
    end
    clf(figure);
end




function error_rate = eval_indices(indices)
    global cheetah_mask;
    global p_prior_fg;
    global p_prior_bg;
    A = makemask(indices);
    
    imagesc(A);
    colormap(gray(255));
    clf(figure);
    
    error = xor(A, cheetah_mask);
    error_rate = sum(sum(error))/(255*270);
    display(error_rate);
    
    error_fg = and(error, cheetah_mask);
    error_bg = and(error, not(cheetah_mask));
    error_rate_fg = sum(sum(error_fg))/sum(sum(cheetah_mask));
    error_rate_bg = sum(sum(error_bg))/sum(sum(not(cheetah_mask)));
    error_rate2 = p_prior_fg*error_rate_fg + p_prior_bg*error_rate_bg;
    display(error_rate_fg);
    display(error_rate_bg);
    display(error_rate2);
end

function A = makemask(indices)
    global ts;
    global cheetah;
    global p_prior_fg;
    global p_prior_bg;
    global mean_fg;
    global mean_bg;
    global var_fg;
    global var_bg;
    global zz;
    
    sigma_fg = cov(ts.TrainsampleDCT_FG);
    sigma_bg = cov(ts.TrainsampleDCT_BG);
    
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
                
                p_cond_fg = multivar_gauss_cov(xdata, mean_fg, sigma_fg, indices);
                p_cond_bg = multivar_gauss_cov(xdata, mean_bg, sigma_bg, indices);
                
                if p_cond_fg * p_prior_fg > p_cond_bg * p_prior_bg
                    A(i, j) = 1;
                end
            end
        end
    end
end

function p = multivar_gauss(block_dct, means, vars, indices)
    p = 1;
    for i = indices
        x = block_dct(i);
        m = means(i);
        v = vars(i);
        p = p * exp(-((x-m).^2)/v)/sqrt(2*pi*v);
    end
end


function p = multivar_gauss_cov(xdata, means, sigma, indices)
    sigma = sigma(indices, indices);
    x = xdata(indices);
    m = means(indices);
    d = abs(det(sigma));
    p = exp(-(x-m) * inv(sigma) * (x-m)'/2 )/sqrt(d*((2*pi)^size(indices,2)));
%     p = mvnpdf(x, m, sigma);
end





