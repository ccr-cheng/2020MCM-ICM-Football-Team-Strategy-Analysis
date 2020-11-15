function [X, Y, oldX] = Model
% 输入数据
% n为数据数，k为变量数
inputTable = readtable('used_stat.csv');
[n, ~] = size(inputTable);
k = 7;
X = zeros(n, k);
X(:,1) = inputTable.Performance1_InvasionIndex;
X(:,2) = inputTable.Performance2_AccelerationIndex;
X(:,3) = inputTable.Performance3_Flexibility;
X(:,4) = inputTable.Performance4_Connectivity;
X(:,5) = inputTable.Performance5_PassingVolume;
X(:,6) = inputTable.Performance6_DefenseIndex;
X(:,7) = inputTable.Performance7_Centrality;
Y = 0.3 * inputTable.Success1_Shots...
    + 0.2 * inputTable.Success2_EfficacyOfShots...
    + 0.3 * inputTable.Success3_Possession...
    + 0.2 * inputTable.Success4_Outcome;

% 删除X异常场次
% X(33,:) = [];
% Y(33,:) = [];
% % n = n - 1;
X(28,:) = [];
Y(28,:) = [];
n = n - 1;
% % % X(27,:) = [];
% % % Y(27,:) = [];
% % % % n = n - 1;
X(16,:) = [];
Y(16,:) = [];
n = n - 1;
% % X(14,:) = [];
% Y(14,:) = [];
% n = n - 1;

% X(6,:) = [];
% Y(6,:) = [];
% n = n - 1;

X = zscore(X);
Y = zscore(Y);
myCorrcoef(X,[],0);

% 主成分分析
Xbar = mean(X);
sX = std(X);
Ybar = mean(Y);
sY = std(Y);
X = zscore(X);
Y = zscore(Y);
[coeff,score,latent,tsquared,explained]= pca(X);
score
coeff
explained
k = 4;
oldX = X;
% X = [score(:,1:k)];
X = [score(:,1:2), score(:,4:5)];
% X = [score(:,1:2), score(:,4)];

% X = [X(:,2), X(:,4), X(:,6)];
% k = k - 4;

X = [ones(n, 1), X];

% 多元线性回归
% a回归系数，aint回归系数置信区间
% r残差即Y - Yhat，rint残差置信区间
alpha = 0.05;
[a, aint, r, rint, stats] = regress(Y, X, alpha);

% score(:,1:k)\Y;
% 原变量回归系数
% b = [coeff(:,1:k)] * a(2:k+1);
% bint = [coeff(:,1:k)] * aint(2:k+1,:);

b = [coeff(:,1:2), coeff(:,4:5)] * a(2:k+1);
bint = [coeff(:,1:2), coeff(:,4:5)] * aint(2:k+1,:);
% b = [Ybar - sY * (Xbar ./ sX) * ab, sY * ab' ./ sX]';

% b = a;
% bint = aint;
 
% 统计量
% R2, F值，F对应的p，残差方差s2=MSE=SSE/(n-k-1)
R2 = stats(1);  % R2
F = stats(2);   % F值
pF = stats(3);  % F值对应的p值
MSE = stats(4); % 残差方差

% 统计量检验
SSE = MSE * (n - k - 1);
SST = std(Y) * std(Y) * (n - 1);
SSR = SST - SSE;
MSR = SSR / k;
% [F, MSR/MSE]
MST = SST / (n - 1);
R2c = 1 - MSE / MST;
% [R2, SSR / SST]
% s2 = sum(r.^2) / (n - k - 1);
% [s2, MSE]

% 回归系数beta显著性检验
A = X' * X;
C = inv(A);
for i = 1 : k + 1
    sa(i) = sqrt(MSE * C(i,i));
    t(i) = a(i) / sa(i);
    pt(i) = 1 - tcdf(t(i), n - k - 1);
end

% 逐步回归法
% 其中的p值是偏回归平方和的F检验的p值
% stepwise(X,Y);
a, aint, b, bint, t = t', pt = pt'
T = table;
T.R2 = R2;
T.R2c = R2c;
T.F = F;
T.pF = pF;
T.MSE = MSE;
T

% MatchID
% Standardized Residual Graphics
Rstd = std(r);
r = r / Rstd;
rint = rint / Rstd;
rcoplot(r, rint);
% 残差-拟合图
% plot(Y - r, r, 'r.', 'Markersize', 20);
end

