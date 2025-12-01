function p = pdf_parzen(X, para)
% Vectorized Parzen window PDF estimation
%  X: test samples (rows = samples, cols = features)
%  para: structure from para_parzen()
%        .labels
%        .samples{c}
%        .parzenw (scalar window width)
%
%  Returns: matrix (rows = samples, cols = classes)

labels = para.labels;
C = length(labels);
[n, d] = size(X);
p = zeros(n, C);

h = para.parzenw;
coef = 1 / ((2*pi)^(d/2) * h^d);

for c = 1:C
    S = para.samples{c};   % training samples for this class
    n_c = size(S,1);
    % Vectorized computation: pairwise distances
    % X: n×d, S: n_c×d → we compute squared distance efficiently
    D2 = pdist2(X, S, 'euclidean').^2;
    % Gaussian kernel sum
    p(:,c) = coef * sum(exp(-0.5 * D2 / (h^2)), 2) / n_c;
end
end

