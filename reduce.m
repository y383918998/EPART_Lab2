function rds = reduce(ds, parts)
% Function reducing samples count of individual classes in ds
% ds - data set to be reduced (sample = row; in the first column labels)
% parts - row vector of reduction coefficients for individual classes
%	(1 means no reduction; 0 means no samples of given class to be left)
% rds - reduced data set

	labels = unique(ds(:, 1));
	if rows(labels) ~= columns(parts)
		error("Class number does not agree with the coefficients number.");
	end

	if max(parts) > 1 || min(parts) < 0
		error("Invalid reduction coefficients.");
	end

	rds = [];
	% Reduce the number of samples for each category
	for i = 1:rows(labels)
		cls = labels(i);
		% Extract all samples of this category
		class_samples = ds(ds(:, 1) == cls, :);
		n = rows(class_samples);
		if n == 0
			continue;
		end

		% Calculate the number of samples to be retained
		keep = round(parts(i) * n);
		% Ensure that at least one sample is retained
		if keep < 1 && parts(i) > 0
			keep = 1;
		elseif keep < 1
			keep = 0;
		end

		if keep > 0
			% Randomly shuffle the samples and select the first 'keep' ones.
			perm = randperm(n);
			selected = class_samples(perm(1:keep), :);
			rds = [rds; selected];
		end
	end
end
