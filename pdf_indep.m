function pdf = pdf_indep(pts, para)
% Computes probability density function assuming feature independence
% pts  - contains points for which pdf is computed (sample = row)
% para - structure containing parameters:
%	para.labels - class labels
%	para.mu - features' mean values (row per class)
%	para.sig - features' standard deviations (row per class)
% pdf - probability density matrix
%	row count = number of samples in pts
%	column count = number of classes

	% Initialize the result matrix
	pdf = zeros(rows(pts), rows(para.mu));
	num_classes = rows(para.mu);
	num_features = columns(para.mu);

	% 对每个类别计算概率密度 Calculate the probability density for each category
	for c = 1:num_classes
		% Initialize a one-dimensional density matrix
		onedpdfs = zeros(rows(pts), num_features);

		% Calculate the one-dimensional normal distribution density of each feature
		for f = 1:num_features
			mu = para.mu(c, f);
			sig = para.sig(c, f);
			% Avoid numerical issues caused by a standard deviation of 0
			if sig < 1e-10
				sig = 1e-10;
			end
			onedpdfs(:, f) = normpdf(pts(:, f), mu, sig);
		end

		% Under the assumption of feature independence, the joint probability is the product of the probabilities of each feature.
		pdf(:, c) = prod(onedpdfs, 2);
	end
end
