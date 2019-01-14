clear all;

ts = load('TrainingSamplesDCT_8.mat');
cheetah = im2double(imread('cheetah.bmp'));
cheetah_mask = im2double(imread('cheetah_mask.bmp'));
zz = load('Zig-Zag Pattern.txt');

nsample_fg = size(ts.TrainsampleDCT_FG, 1);
nsample_bg = size(ts.TrainsampleDCT_BG, 1);
p_prior_fg = nsample_fg/(nsample_bg + nsample_fg);
p_prior_bg = nsample_bg/(nsample_bg + nsample_fg);

display(p_prior_fg);
display(p_prior_bg);

train_result_fg = train(ts.TrainsampleDCT_FG, nsample_fg);
train_result_bg = train(ts.TrainsampleDCT_BG, nsample_bg);
figure;
hold on;
hist_bg = histogram(train_result_bg, 0:64,'facecolor','r');
hist_fg = histogram(train_result_fg, 0:64,'facecolor','b');
legend('grass', 'cheetah');
xlabel('Index');
ylabel('Frequency of the 2nd largest index');
clf(figure);

cond_fg = hist_fg.Values/nsample_fg;
cond_bg = hist_bg.Values/nsample_bg;

hold on;
bar(cond_bg,1.0,'facecolor','r', 'FaceAlpha', 0.5);
bar(cond_fg,1.0,'facecolor','b', 'FaceAlpha', 0.5);
legend('P(X=x|Y=grass)', 'P(X=x|Y=cheetah)');
xlabel('Index');
ylabel('Conditional Probability');
xlim([0 64]);
clf(figure);

hold on;
bar(cond_fg,1.0);
legend('P(X=x|Y=cheetah)');
xlabel('Index');
ylabel('Conditional Probability');
xlim([0 64]);
ylim([0 0.5]);
clf(figure);

hold on;
bar(cond_bg,1.0);
legend('P(X=x|Y=grass)');
xlabel('Index');
ylabel('Conditional Probability');
xlim([0 64]);
ylim([0 0.5]);
clf(figure);


X = zeros(size(cheetah));
A = zeros(size(cheetah));
for i = 1:size(cheetah, 1)
    for j = 1:size(cheetah, 2)
        if 5<i && i<size(cheetah, 1)-4 && 5<j && j<size(cheetah, 2)-4
            block = cheetah(i-4:i+3, j-4:j+3);
            block_dct = abs(dct2(block, 8, 8));
            [~, maxidx] = max(block_dct(:));
            [maxidx_row, maxidx_col] = ind2sub(size(block_dct),maxidx);
            block_dct(maxidx_row, maxidx_col) = 0;
            [~, maxidx] = max(block_dct(:));
            [maxidx_row, maxidx_col] = ind2sub(size(block_dct),maxidx);
            x = zz(maxidx_row, maxidx_col);
            X(i, j) = x;
            if cond_fg(x+1)*p_prior_fg > cond_bg(x+1)*p_prior_bg
                A(i, j) = 1;
            end
        end
    end
end

imagesc(X);
colormap(gray(255));
clf(figure);

imagesc(A);
colormap(gray(255));

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



function train_result = train(trainsample, nsample)
    train_result = zeros(nsample(1), 1);
    for i = 1:nsample
        data = trainsample(i, :);
        [~, maxidx] = max(data(:));
        data(maxidx) = 0;
        [~, maxidx] = max(data(:));
        train_result(i) = maxidx - 1;
    end
end

