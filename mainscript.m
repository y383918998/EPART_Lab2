% tiny data file to verify pdf functions成功!!!
load pdf_test.txt
size(pdf_test)

% how many classes are there?成功!!!
labels = unique(pdf_test(:,1))

% how many samples are in each class?成功!!!
[labels'; sum(pdf_test(:,1) == labels')]
		  % ^^^ how does this expression work?

% what's the layout of the samples?
% will it work?成功!!!
plot2features(pdf_test, 2, 3)

% check if statistics package is present成功!!!
normpdf(0, 0, 1)
% it can work directly - nothing to be done 
% it can be installed but not loaded - pkg load statistics
% it can be not installed at all - use __normpdf function provided instead


pdfindep_para = para_indep(pdf_test)
% para_indep indep is already implemented; it should give:

% pdfindep_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%       0.21772   0.19172
%       0.19087   0.27179

% now you have to implement pdf_indep and then verify it

pi_pdf = pdf_indep(pdf_test([2 7 12 17],2:end), pdfindep_para)
%成功
%pi_pdf =
%  1.4700e+000  4.5476e-007
%  3.4621e+000  4.9711e-005
%  6.7800e-011  2.7920e-001
%  5.6610e-008  1.8097e+000

% multivariate normal distribution - parameters ...

pdfmulti_para = para_multi(pdf_test)
%成功
%pdfmulti_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%    ans(:,:,1) =
%       0.047401   0.018222
%       0.018222   0.036756
%    ans(:,:,2) =
%       0.036432  -0.033186
%      -0.033186   0.073868  

% ... and probability density function (use mvnpdf in pdf_multi)

pm_pdf = pdf_multi(pdf_test([2 7 12 17],2:end), pdfmulti_para)
%成功
%pm_pdf =
%  7.9450e-001  6.5308e-017
%  3.9535e+000  3.8239e-013
%  1.6357e-009  8.6220e-001
%  4.5833e-006  2.8928e+000

% parameters for Parzen window approximation 
pdfparzen_para = para_parzen(pdf_test, 0.5)
									 % ^^^ window width

%pdfparzen_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    samples =
%    {
%      [1,1] =
%         1.10000   0.95000
%         0.98000   0.61000
% .....
%         0.69000   0.93000
%         0.79000   1.01000
%      [2,1] =
%        -0.010000   0.380000
%         0.250000  -0.440000
% .....
%        -0.110000   0.030000
%         0.120000  -0.090000
%    }
%    parzenw =  0.50000

% now you have to implement pdf_parzen and then verify it

pp_pdf = pdf_parzen(pdf_test([2 7 12 17],2:end), pdfparzen_para)

%pp_pdf =
%  9.7779e-001  6.1499e-008
%  2.1351e+000  4.2542e-006
%  9.4059e-010  9.8823e-001
%  2.0439e-006  1.9815e+000


% now you can start work with cards!
[train test] = load_cardsuits_data();

% Our first look at the data
size(train)
size(test)
labels = unique(train(:,1))
unique(test(:,1))
[labels'; sum(train(:,1) == labels')]

% the first task after loading the data is checking
% training set for outliers; to this end we usually compute 
% simple statistics: mean, median, std, 
% and/or plot histogram of individual feature: hist
% and/or plot two features at a time: plot2features

[mean(train); median(train)]
hist(train(:,1))
plot2features(train, 4, 6)
					%^^^^ just an example

% to identify outliers you can use two output argument versions 
% of min and max functions

[mv midx] = max(train)
% 186 is the index of the identifier 可以从最大值筛选看出186行有问题

train(midx-2:midx+2, :)
% 细分从184到188行可看出，186行确实数据异常


% sample 186 is suspected


% theere were huge difference in all features (the label was the )
% it seems that these three rows are very similar to each other...
% that's because 41 is evidently not an outlier index

% if you're sure tha midx sample should be removed:
size(train)
train(midx, :) = [];
size(train)


%after displayonh plot of the first two features anotheor outlayer is visiable
[mv midx] = min(train)
%641 was identified as possible outlier
midx = 641
train(midx-2:midx+2, :)

%after set midx = 641 .dont forget to reset the train!!!!!别忘了重置train，在设置midx为特定值
train(midx, :) = [];
size(train)
% the procedure of searching for and removing outliers must be repeated 
% until no outliers exist in the training set

% after removing outliers, you can deal with the selection of TWO features for classification
% in this case, it is enough to look at the graphs of two features and choose the ones that
% give relatively well separated classes

% 我会选择3＆4图
% after selecting features reduce both sets:别忘了重置，如下所示
train = train(:, [1 3 4]);
test = test(:, [1 3 4]);
				% ^^^ please, don't use these features!
%用size(train)和size(test)检查结果是否相似或相同，验证重置成功。

% POINT 2

[train, test] = load_cardsuits_data();   % 重载一次干净数据
train([186 641], :) = [];                % 删除异常点

train = train(:, [1 3 4]);
test  = test(:, [1 3 4]);

size(train)
size(test)
%  Result：train ≈ 1822×3  test ≈ 1824×3

pdfindep_para = para_indep(train);
pdfmulti_para = para_multi(train);
pdfparzen_para = para_parzen(train, 0.001); 
% this window width should be included in your report!

%pdfindep_para查看

% Point 2 results
base_ercf = zeros(1,3);
base_ercf(1) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
base_ercf(2) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
base_ercf(3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
base_ercf


% base_ercf = 0.021382   0.020833   0.016996    结果所示
% before moving to point 3 it would be wise to
% implement and test reduce function
% let's start with small test set - just 2 classes

rdlab = unique(pdf_test(:,1));
reduced = reduce(pdf_test, [0.8 0.4]);
[rdlab'; sum(reduced(:,1) == rdlab')]

% ans =
%     1    2
%     8    4





% POINT 3

% In the next point, the reduce function will be useful, which reduces the number of samples 
% in the individual classes (in this case, the reduction will be the same in all classes - 
% OF THE TRAINING SET)
% Because reduce has to draw samples randomly, the experiment should be repeated 5 times
% In the report, please provide only the mean value and the standard deviation 
% of the error coefficient

% YOUR CODE GOES HERE 
%
% ==== Point 3: Reduced Training Sets ====

parts = [0.1 0.25 0.5];  % three reduction levels
rep_cnt = 5;             % repeat each experiment 5 times
class_count = length(unique(train(:,1)));  % number of classes (should be 8)

% to store results: rows = 3 reduction levels, columns = 3 classifiers
err_means = zeros(length(parts), 3);
err_stds  = zeros(length(parts), 3);

apriori = 0.125 * ones(1, class_count);

for p = 1:length(parts)
    errs = zeros(rep_cnt, 3);  % each classifier’s error per repetition

    for r = 1:rep_cnt
        % randomly reduce training set
        reduced_train = reduce(train, parts(p) * ones(1, class_count));

        % estimate parameters for each model
        pdfindep_para = para_indep(reduced_train);
        pdfmulti_para = para_multi(reduced_train);
        pdfparzen_para = para_parzen(reduced_train, 0.001);

        % compute classification errors on the same test set
        errs(r,1) = mean(bayescls(test(:,2:end), @pdf_indep,  pdfindep_para,  apriori) != test(:,1));
        errs(r,2) = mean(bayescls(test(:,2:end), @pdf_multi,  pdfmulti_para,  apriori) != test(:,1));
        errs(r,3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para, apriori) != test(:,1));
    end

    % summarize
    err_means(p,:) = mean(errs);
    err_stds(p,:)  = std(errs);
end

% display results as table
disp('Mean classification errors for reduced training sets:')
disp(err_means)
disp('Standard deviations:')
disp(err_stds)

% note that for given experiment you should reduce all classes in the training
% set with the same reduction coefficient; assuming that class_count is the 
% number of different classes in the training set you can take 3/4 random samples
% of each class with:
% 	reduced_train = reduce(train, 0.75 * ones(1, class_count))
%

%          “Effect of training set reduction on Bayesian classifier accuracy”
figure;
plot(parts, err_means(:,1), '-o', 'DisplayName', 'Independent Gaussian'); hold on;
plot(parts, err_means(:,2), '-s', 'DisplayName', 'Multivariate Gaussian');
plot(parts, err_means(:,3), '-d', 'DisplayName', 'Parzen window');
xlabel('Training set reduction ratio');
ylabel('Mean classification error');
legend('Location','northeast');
title('Effect of Training Set Reduction on Error Rate');
grid on;






% POINT 4
% Point 4 concerns only Parzen window classifier (on the full training set)

parzen_widths = [0.0001, 0.0005, 0.001, 0.005, 0.01];
parzen_res = zeros(1, columns(parzen_widths));

% YOUR CODE GOES HERE 
%
% ==== Point 4: Parzen window width sweep ====

% 1. Different bandwidths we will test 我们要测试的不同带宽
parzen_widths = [0.0001, 0.0005, 0.001, 0.005, 0.01];

% 2. To store the classification error rate for each bandwidth 用来存每个带宽下的分类错误率
parzen_res = zeros(1, columns(parzen_widths));

% 3. Prior probability: 8 classes equally likely = 0.125 先验概率：8类均等 = 0.125
class_count = length(unique(train(:,1)));
apriori = 0.125 * ones(1, class_count);

% 4. Perform a complete evaluation for each bandwidth 对每一个带宽做一次完整评估
for i = 1:columns(parzen_widths)

    h = parzen_widths(i);

    % Estimate Parzen parameters using the current bandwidth 用当前带宽估计Parzen参数
    pdfparzen_para = para_parzen(train, h);

    % Classify on the test set using this Parzen classifier 用这个Parzen分类器在测试集上分类
    parzen_res(i) = mean( ...
        bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para, apriori) ...
        != test(:,1) ...
    );
end

% 5. Print the results, showing the error rate for each bandwidth row by row 打印结果，逐行看每个带宽的错误率
[parzen_widths; parzen_res]

% 6. Plot the results (X-axis = Bandwidth, Y-axis = Error Rate; use a log scale for better visualization) 画图（横轴=带宽，纵轴=错误率，使用对数坐标更直观）
figure;
semilogx(parzen_widths, parzen_res, '-o');
xlabel('Parzen window width (h)');
ylabel('classification error');
title('Effect of Parzen window width');
grid on;

[parzen_widths; parzen_res]

% Plots are sometimes better than numerical results
semilogx(parzen_widths, parzen_res)




% POINT 5
% In point 5 you should reduce TEST SET
%
 
apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];
parts = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];

% YOUR CODE GOES HERE 
%
% ==== Point 5: Imbalanced test set and non-uniform priors ====

% 1. Reprepare training/test data 重新准备训练/测试（以防你之前清空了环境）
[train_full, test_full] = load_cardsuits_data();
% Remove outliers 去掉离群点
train_full([186 641], :) = [];
% Only keep the label + feature 3 + feature 4 只保留标签 + 特征3 + 特征4
train = train_full(:, [1 3 4]);
test  = test_full(:,  [1 3 4]);

% 2. Train/Estimate the parameters for the three classifiers (using the full training set) 训练/估计三个分类器的参数（用完整训练集）
pdfindep_para  = para_indep(train);
pdfmulti_para  = para_multi(train);
pdfparzen_para = para_parzen(train, 0.001);

% 3. Define the prior probabilities (non-uniform) 定义先验概率 (非均匀)
apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];

% 4. Define the retention proportions for the test set 定义针对测试集的保持比例
parts = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];

% 5. Use reduce() to generate a "biased test set" 用 reduce() 生成“偏置测试集”（某些类保留较少样本）
biased_test = reduce(test, parts);

% 6. Calculate the error rates of the three classifiers on the biased test set 在偏置测试集上计算三种分类器的错误率
err_indep_biased = mean(bayescls(biased_test(:,2:end), @pdf_indep,  pdfindep_para,  apriori) != biased_test(:,1));
err_multi_biased = mean(bayescls(biased_test(:,2:end), @pdf_multi,  pdfmulti_para,  apriori) != biased_test(:,1));
err_parzen_biased= mean(bayescls(biased_test(:,2:end), @pdf_parzen, pdfparzen_para, apriori) != biased_test(:,1));

[err_indep_biased err_multi_biased err_parzen_biased]

% 7. Additionally, check how many samples remain in each class of the biased test set, for easier report writing 另外：看一下偏置测试集每类还剩多少样本，方便你写报告
labs = unique(biased_test(:,1));
[labs'; sum(biased_test(:,1) == labs')]





% =================POINT 6=======================
% Please, use the same features as in Bayes classification experiments
% In point 6 we should consider data normalization

std(train(:,2:end))

% Should we normalize?
% If YES remember to normalize BOTH training and testing sets

% YOUR CODE GOES HERE 
%
% ==== Point 6: 1-NN baseline (no normalization) ====

%或者是以下代码，结果一样的。第一步：
[train_full, test_full] = load_cardsuits_data();
train_full([186 641], :) = [];      % remove outliers
train = train_full(:, [1 3 4]);     % keep label, feature3, feature4
test  = test_full(:,  [1 3 4]);

nn_pred = zeros(rows(test),1);
for i = 1:rows(test)
    nn_pred(i) = cls1nn(test(i,2:end), train);
end

nn_err_raw = mean(nn_pred != test(:,1))

%第二部：
mu  = mean(train(:,2:end));     % 1x2 mean of feature3,4
sig = std(train(:,2:end));      % 1x2 std  of feature3,4

train_norm = train;
test_norm  = test;

train_norm(:,2:end) = (train(:,2:end) - mu) ./ sig;
test_norm(:,2:end)  = (test(:,2:end)  - mu) ./ sig;

nn_pred_norm = zeros(rows(test_norm),1);
for i = 1:rows(test_norm)
    nn_pred_norm(i) = cls1nn(test_norm(i,2:end), train_norm);
end

nn_err_norm = mean(nn_pred_norm != test_norm(:,1))
%以上两个方法，两步输出结果为：nn_err_raw = 0.018092   &    nn_err_norm = 0.018092








% % Step A. 重新准备干净数据（防止你清空过workspace）
% [train_full, test_full] = load_cardsuits_data();
% train_full([186 641], :) = [];          % 去除离群点
% train = train_full(:, [1 3 4]);         % 选两维特征
% test  = test_full(:,  [1 3 4]);

% % Step B. 逐个测试样本用1-NN分类
% nn_pred = zeros(rows(test),1);
% for i = 1:rows(test)
%     nn_pred(i) = cls1nn(test(i,2:end), train);  % test(i,2:end) 是没有标签的特征向量
% end

% % Step C. 计算错误率
% nn_err_raw = mean(nn_pred != test(:,1))

% nn_err_raw

% % ==== Point 6: 1-NN with standardization ====

% % Step D. 计算训练集特征(列2和3)的均值和标准差
% mu  = mean(train(:,2:end));     % 1x2
% sig = std(train(:,2:end));      % 1x2

% % Step E. 用训练集的mu和sig标准化训练集和测试集
% train_norm = train;
% test_norm  = test;

% train_norm(:,2:end) = (train(:,2:end) - mu) ./ sig;
% test_norm(:,2:end)  = (test(:,2:end)  - mu) ./ sig;  % 注意这里也是用train的mu和sig

% % Step F. 再做一次1-NN（这次用归一化后的特征）
% nn_pred_norm = zeros(rows(test_norm),1);
% for i = 1:rows(test_norm)
%     nn_pred_norm(i) = cls1nn(test_norm(i,2:end), train_norm);
% end

% % Step G. 计算标准化后的1-NN错误率
% nn_err_norm = mean(nn_pred_norm != test_norm(:,1))

% nn_err_norm
