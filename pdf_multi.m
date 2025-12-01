function pdf = pdf_multi(pts, para)
% Computes multivariate probability density function
% pts  - contains points for which pdf is computed (sample = row)
% para - structure containing parameters:
%	para.labels - class labels
%	para.mu - features' mean values (row per class)
%	para.sig - features' covariance matrices (LAYER per class)
% pdf - probability density matrix
%	row count = number of samples in pts
%	column count = number of classes

	% Initialize the result matrix
	pdf = zeros(rows(pts), rows(para.mu));
	num_classes = rows(para.mu);
	num_features = columns(para.mu);

	% Calculate the multidimensional Gaussian distribution density for each category
	for c = 1:num_classes
		mu = para.mu(c, :);
		sigma = para.sig(:, :, c);

		% Regularize the covariance matrix to prevent singularity
		epsilon = 1e-6;
		sigma = sigma + epsilon * eye(size(sigma));

		% Calculate the multivariate normal distribution density
		pdf(:, c) = mvnpdf(pts, mu, sigma);
	end
end
