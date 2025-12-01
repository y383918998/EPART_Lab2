% mainscript.m — EPART_L2 reporting runner
clear; clc; close all;

% ---------- I/O ----------
outdir = 'report_outputs';
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ---------- sanity: stats package (Octave) ----------
try, normpdf(0,0,1); catch, try, pkg load statistics; end; end

% ---------- load data ----------
[train_raw, test_raw] = load_cardsuits_data();

% ======================Point 1: basic stats, cleansing, feature choice ==============================
disp('--- POINT 1: Data inspection and cleansing ---');

[train, test] = load_cardsuits_data();

disp('Size of train and test sets:');
disp(size(train));
disp(size(test));

labels = unique(train(:,1));
disp('Class labels and counts in train set:');
disp([labels'; sum(train(:,1) == labels')]);
csvwrite(fullfile(outdir,'point1_class_counts_train.csv'), ...
         [labels'; sum(train(:,1) == labels')]);

% ---- Basic statistics before cleansing ----
disp('Mean and median values before cleansing:');

stats_before = [mean(train); median(train)];
disp(stats_before);
csvwrite(fullfile(outdir,'train_mean_before.csv'), mean(train));
csvwrite(fullfile(outdir,'train_median_before.csv'), median(train));

% ---- Visualization before cleansing ----
plot2features(train, 3, 7);
title('Features 3 vs 7 before cleansing');
print(fullfile(outdir,'point1_before_scatter.png'), '-dpng', '-r150');
close(gcf);

% ---- Remove potential max/min outliers ----
[maxVals, maxIdx] = max(train);
minIdx = maxIdx(2);
train(minIdx,:) = [];

[minVals, minIdx] = min(train);
minIdx = minIdx(2);
train(minIdx,:) = [];

% ---- Visualization after cleansing ----
plot2features(train, 3, 7);
title('Features 3 vs 7 after cleansing');
print(fullfile(outdir,'point1_after_scatter.png'), '-dpng', '-r150');
close(gcf);

% ---- Basic statistics after cleansing ----
disp('Mean and median values after cleansing:');
stats_after = [mean(train); median(train)];
disp(stats_after);
csvwrite(fullfile(outdir,'train_mean_after.csv'), mean(train));
csvwrite(fullfile(outdir,'train_median_after.csv'), median(train));

% ---- Feature selection ----
chosenFeatures = [3 7];
train = train(:, [1 chosenFeatures]);
test  = test(:,  [1 chosenFeatures]);

fid = fopen(fullfile(outdir,'point1_chosen_features.txt'),'w');
fprintf(fid,'Chosen features: [%d %d]\n', chosenFeatures(1), chosenFeatures(2));
fclose(fid);

disp('--- POINT 1 done ---');


% ---------- Point 2: equal priors 0.125×8 ----------
apr = 0.125 * ones(1,8);
p_ind = para_indep(train);
p_mul = para_multi(train);
p_par = para_parzen(train, 0.001); % include h1 in report

e2_ind = mean(bayescls(test(:,2:end), @pdf_indep,  p_ind, apr) ~= test(:,1));
e2_mul = mean(bayescls(test(:,2:end), @pdf_multi,  p_mul, apr) ~= test(:,1));
e2_par = mean(bayescls(test(:,2:end), @pdf_parzen, p_par, apr) ~= test(:,1));
e2 = [e2_ind, e2_mul, e2_par];
csvwrite(fullfile(outdir,'point2_equal_priors_errors.csv'), e2);

% ---------- Point 3: reduce TRAIN equally; repeat; mean/std ----------
parts = [0.1 0.25 0.5];
rep_cnt = 5;
C = length(unique(train(:,1)));

E = zeros(3, numel(parts), rep_cnt); % methods × parts × reps
for k = 1:numel(parts)
  for r = 1:rep_cnt
    tr_red = reduce(train, parts(k) * ones(1,C));
    pi_r = para_indep(tr_red);
    pm_r = para_multi(tr_red);
    pp_r = para_parzen(tr_red, 0.001);
    E(1,k,r) = mean(bayescls(test(:,2:end), @pdf_indep,  pi_r, apr) ~= test(:,1));
    E(2,k,r) = mean(bayescls(test(:,2:end), @pdf_multi,  pm_r, apr) ~= test(:,1));
    E(3,k,r) = mean(bayescls(test(:,2:end), @pdf_parzen, pp_r, apr) ~= test(:,1));
  end
end
Emean = mean(E,3); Estd = std(E,0,3);
Emin  = min(E,[],3); Emax = max(E,[],3);
csvwrite(fullfile(outdir,'point3_mean.csv'), Emean);
csvwrite(fullfile(outdir,'point3_std.csv'),  Estd);
csvwrite(fullfile(outdir,'point3_min.csv'),  Emin);
csvwrite(fullfile(outdir,'point3_max.csv'),  Emax);

% ---------- Point 4: Parzen width sweep on full train ----------
pw = [1e-4 5e-4 1e-3 5e-3 1e-2];
err_pw = zeros(size(pw));
for i = 1:numel(pw)
  pp = para_parzen(train, pw(i));
  err_pw(i) = mean(bayescls(test(:,2:end), @pdf_parzen, pp, apr) ~= test(:,1));
end
dlmwrite(fullfile(outdir,'point4_parzen_scan.csv'), [pw(:), err_pw(:)], ',');
figure; semilogx(pw, err_pw, '-o'); grid on;
xlabel('Parzen h_1 (log)'); ylabel('Error rate'); title('Parzen width vs error');
print(fullfile(outdir,'point4_parzen_scan.png'), '-dpng','-r150');

% ---------- Point 5: modify priors + reduce TEST (red suits) ----------
% note: ensure mapping matches your dataset; adjust if needed
apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];
parts_test = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];
test_p5 = reduce(test, parts_test);

pi5 = para_indep(train);
pm5 = para_multi(train);
pp5 = para_parzen(train, 0.001);

e5_ind = mean(bayescls(test_p5(:,2:end), @pdf_indep,  pi5, apriori) ~= test_p5(:,1));
e5_mul = mean(bayescls(test_p5(:,2:end), @pdf_multi,  pm5, apriori) ~= test_p5(:,1));
e5_par = mean(bayescls(test_p5(:,2:end), @pdf_parzen, pp5, apriori) ~= test_p5(:,1));
csvwrite(fullfile(outdir,'point5_changed_priors_errors.csv'), [e5_ind, e5_mul, e5_par]);
csvwrite(fullfile(outdir,'point5_unchanged_priors_errors.csv'), e2); % baseline from Point 2

% ---------- Point 6: 1-NN before/after z-score ----------
err1_raw = mean(cls1nn(test(:,2:end), train) ~= test(:,1));
mu_tr = mean(train(:,2:end)); sg_tr = std(train(:,2:end)); sg_tr(sg_tr<1e-10)=1e-10;
trN = train; teN = test;
trN(:,2:end) = (train(:,2:end)-mu_tr)./sg_tr;
teN(:,2:end) = (test(:,2:end)-mu_tr)./sg_tr;
err1_z = mean(cls1nn(teN(:,2:end), trN) ~= teN(:,1));
csvwrite(fullfile(outdir,'point6_1nn_errors.csv'), [err1_raw, err1_z]);

% ---------- summary ----------
fid = fopen(fullfile(outdir,'summary.txt'),'w');
fprintf(fid,'Features used: [%d %d]\n', chosenFeatures(1), chosenFeatures(2));
fprintf(fid,'Point2 errors [ind, mul, par]: %s\n', mat2str(e2,5));
fprintf(fid,'Point3 mean errors (rows methods 1..3, cols parts):\n%s\n', mat2str(Emean,5));
fprintf(fid,'Point4 widths/errors:\n%s\n', mat2str([pw; err_pw],5));
fprintf(fid,'Point5 changed priors errors [ind,mul,par]: %s\n', mat2str([e5_ind,e5_mul,e5_par],5));
fprintf(fid,'Point6 1-NN [raw,zscore]: %s\n', mat2str([err1_raw, err1_z],5));
fclose(fid);

disp('Done. See report_outputs/ for CSVs and figures.');

