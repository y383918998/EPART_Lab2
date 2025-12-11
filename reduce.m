function rds = reduce(ds, parts)
% Function reducing samples count of individual classes in ds
% ds - data set to be reduced (sample = row; in the first column labels)
% parts - row vector of reduction coefficients for individual classes
%	(1 means no reduction; 0 means no samples of given class to be left)
% rds - reduced data set

	labels = unique(ds(:,1));
	if rows(labels) ~= columns(parts)
		error("Class number does not agree with the coefficients number.");
	end

	if max(parts) > 1 || min(parts) < 0
		error("Invalid reduction coefficients.");
	end

	% YOUR CODE GOES HERE
	
	rds = [];
	% for each class
	for clid = 1: rows(labels)
		% select only one class samples from ds
		aclass = ds(ds(:, 1) == labels(clid, 1), :);
		% shuffle samples of this class with randperm
		% select proper part of shuffled class and append it to rds
		sample_count = int64(rows(aclass) * parts(clid));
		rds = [rds; aclass(randperm(rows(aclass), sample_count), :)];
    end
end
