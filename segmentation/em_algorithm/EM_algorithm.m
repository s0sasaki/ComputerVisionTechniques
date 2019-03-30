clear all;
global cheetah;
global cheetah_mask;
global p_prior_fg;
global p_prior_bg;
global zz;

ts = load('TrainingSamplesDCT_8_new.mat');
cheetah = im2double(imread('cheetah.bmp'));
cheetah_mask = im2double(imread('cheetah_mask.bmp'));
zz = load('Zig-Zag Pattern.txt');

nsample_fg = size(ts.TrainsampleDCT_FG, 1);
nsample_bg = size(ts.TrainsampleDCT_BG, 1);
p_prior_fg = nsample_fg/(nsample_bg + nsample_fg);
p_prior_bg = nsample_bg/(nsample_bg + nsample_fg);



% %%%% Q1
% for c = [8]
%     c
%     k_params = [1 2 4 8 16 24 32 40 48 56 64];
%     n_experiments = 5*5;
%     results = zeros(n_experiments, size(k_params,2));
%     for i = 1:n_experiments
%         disp(i);
%         error_list = [];
%         for k = k_params
%             [pic_fg, muc_fg, sigmac_fg] = em(ts.TrainsampleDCT_FG(:, 1:k), c, k);
%             [pic_bg, muc_bg, sigmac_bg] = em(ts.TrainsampleDCT_BG(:, 1:k), c, k);
%             A = makemask(c, k, pic_fg, pic_bg, muc_fg, muc_bg, sigmac_fg, sigmac_bg);
%             error = eval_mask(A);
%             error_list = [error_list error];
%         end
%         disp(error); %monitor
%         results(i,:) = error_list;
%     end
%     plot(results');
%     xticklabels(k_params);
%     title(strcat('c=', int2str(c)));
%     xlabel('x');
%     ylabel('error rate');
%     saveas(gcf, strcat('q1_c', int2str(c), '.png'));
%     csvwrite(strcat('q1_c', int2str(c), '.csv'), results');
%     clf(figure);
%     results_mean = mean(results, 1);
%     plot(results_mean);
%     xticklabels(k_params);
%     title(strcat('c=', int2str(c)));
%     xlabel('x');
%     ylabel('error rate');
%     saveas(gcf, strcat('q1_c', int2str(c), '_mean.png'));
%     clf(figure);
% end

%%%% Q2
c_list = [1 2 4 8 16 32];
k_params = [1 2 4 8 16 24 32 40 48 56 64];
results = zeros(size(c_list, 2), size(k_params,2));
for j = 1:size(c_list, 2)
    c = c_list(j);
    disp(c);
    for t = 1:size(k_params, 2)
        k = k_params(t);
        [pic_fg, muc_fg, sigmac_fg] = em(ts.TrainsampleDCT_FG(:, 1:k), c, k);
        [pic_bg, muc_bg, sigmac_bg] = em(ts.TrainsampleDCT_BG(:, 1:k), c, k);
        A = makemask(c, k, pic_fg, pic_bg, muc_fg, muc_bg, sigmac_fg, sigmac_bg);
        error = eval_mask(A);
        results(j,t) = error;
    end
end
plot(results'); 
xticklabels(k_params);
legend('c=1', 'c=2', 'c=4', 'c=8', 'c=16', 'c=32');
xlabel('x');
ylabel('error rate');
saveas(gcf, 'q2.png');
csvwrite('q2.csv', results');
clf(figure);

function [pic, muc, sigmac] = em(dctdata, c, k)
    sigmac = zeros(k, k);
    for i = 1:c
        sigmac(:,:,i+1) = eye(k,k);
    end
    sigmac = sigmac(:,:,2:end);
    muc = rand(1, k, c);
    pic = rand(1, c);
    pic = pic./sum(pic);
    h = zeros(size(dctdata, 1), c);
    for tmp = 1:50 %4
        % E step ------------------------
        % stop if sigma does not have proper values
        flag = true;
        for j = 1:c
            flag = flag && all(eig(sigmac(:,:,j))>=0);
            flag = flag && max(eig(sigmac(:,:,j)))>=0.001; %or we can use a lower bound
        end
        if not(flag)
            break;
        end
        
        for i = 1:size(dctdata, 1)
            for j = 1:c
                h(i,j) = pic(1,j) * mvnpdf(dctdata(i,:), muc(:,:,j), sigmac(:,:,j)); 
            end
        end
        h = h./sum(h, 2);
        
        % M step ------------------------
        eig_sigma_tmp = sum(((dctdata- muc(:,:,j)).^2) .* h(:,j), 1) ./ sum(h(:,j), 1);
        
        % stop if sigma does not have proper values
        flag = true;
        for j = 1:c
            flag = flag && all(eig_sigma_tmp >=0);
            flag = flag && max(eig_sigma_tmp)>=0.001; %or we can use a lower bound
        end
        if not(flag)
            break;
        end
        
        pic = sum(h, 1)/size(dctdata, 1);
        for j = 1:c
            eig_sigma_tmp = sum(((dctdata- muc(:,:,j)).^2) .* h(:,j), 1) ./ sum(h(:,j), 1);
            eig_sigma_tmp = max(eig_sigma_tmp, 0.001);
            sigmac(:,:,j) = diag(eig_sigma_tmp);
            %sigmac(:,:,j) = diag(sum(((dctdata- muc(:,:,j)).^2) .* h(:,j), 1) ./ sum(h(:,j), 1));
            muc(:,:,j) = sum(dctdata .* h(:,j), 1) ./ sum(h(:,j), 1);
        end
    end
end

function p = mix_mvnpdf(xdata, c, pic, muc, sigmac)
    p = 0;
    for i = 1:c
        p = p + pic(i) * mvnpdf(xdata, muc(:, :, i), sigmac(:, :, i));
    end
end

function A = makemask(c, k, pic_fg, pic_bg, muc_fg, muc_bg, sigmac_fg, sigmac_bg)
    global cheetah;
    global p_prior_fg;
    global p_prior_bg;
    global zz;
    
    for j = 1:c
        diag_sigmac_fg = diag(sigmac_fg(:,:,j));
        diag_sigmac_bg = diag(sigmac_bg(:,:,j));
        diag_sigmac_fg = max(diag_sigmac_fg, 0.001);
        diag_sigmac_bg = max(diag_sigmac_bg, 0.001);
        sigmac_fg(:,:,j) = diag(diag_sigmac_fg);
        sigmac_bg(:,:,j) = diag(diag_sigmac_bg);
    end
    
    A = zeros(size(cheetah));
    for i = 1:size(cheetah, 1)
        for j = 1:size(cheetah, 2)
            if 5<i && i<size(cheetah, 1)-4 && 5<j && j<size(cheetah, 2)-4
                block = cheetah(i-4:i+3, j-4:j+3);
                block_dct = abs(dct2(block, 8, 8));
                xdata = zeros(1, 64);
                xdata(zz+1) = block_dct;
                
                % this is for the code of Q1(old)
%                 size(pic_fg)
%                 size(muc_fg)
%                 size(sigmac_fg)
%                 p_cond_fg = mix_mvnpdf(xdata(1:k), c, pic_fg(1,1:k,:), muc_fg(1,1:k,:), sigmac_fg(1:k,1:k,:)); 
%                 p_cond_bg = mix_mvnpdf(xdata(1:k), c, pic_bg(1,1:k,:), muc_bg(1,1:k,:), sigmac_bg(1:k,1:k,:));
                p_cond_fg = mix_mvnpdf(xdata(1:k), c, pic_fg, muc_fg, sigmac_fg); 
                p_cond_bg = mix_mvnpdf(xdata(1:k), c, pic_bg, muc_bg, sigmac_bg);
                
                if p_cond_fg * p_prior_fg > p_cond_bg * p_prior_bg
                    A(i, j) = 1;
                end
            end
        end
    end
end

function error_rate = eval_mask(A)
    global cheetah_mask;
    
    %imagesc(A);
    %colormap(gray(255));
    %clf(figure);

    error = xor(A, cheetah_mask);
    error_rate = sum(sum(error))/(255*270);
    %disp(error_rate);
end

