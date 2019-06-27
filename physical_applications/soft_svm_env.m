%% RUN SVM
% Visible Tests
rng(15)

X = [randn(20,2);randn(20,2)+4];
t = [repmat(-1,20,1); repmat(1,20,1)];
% Add a bad point
X = [X;[1 1]];
t = [t;1];

[alpha, bias] = softsvm(X, t, .01);
warning off
ma = {'ko','ks'};
fc = {[0 0 0],[1 1 1]};
tv = unique(t);
figure(1);hold off
pos = find(alpha>1e-6);
plot(X(pos,1),X(pos,2),'ko','markersize',15,'markerfacecolor',[0.6 0.6 0.6],...
    'markeredgecolor',[0.6 0.6 0.6]);
hold on
for i = 1:length(tv)
    pos = find(t==tv(i));
    plot(X(pos,1),X(pos,2),ma{i},'markerfacecolor',fc{i});
end

xp = xlim;
yl = ylim;
% Because this is a linear SVM, we can compute w and plot the decision
% boundary exactly.
w = sum(repmat(alpha.*t,1,2).*X,1)';
w = w./norm(w)
w_correct = [.6340,.7733];


yp = -(bias + w(1)*xp)/w(2);
figure(1)
plot(xp,yp,'k','linewidth',2);
ylim(yl);
title('Your Softmargin SVM');
