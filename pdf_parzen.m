function pdf = pdf_parzen(pts, para)
% Approximates probability density function with Parzen window
% pts  - contains points for which pdf is computed (sample = row)
% para - structure containing parameters:
%	para.labels - class labels
%	para.samples - cell array containing class samples
%	para.parzenw - Parzen window width
% pdf - probability density matrix
%	row count = number of samples in pts
%	column count = number of classes

	% final result matrix
	pdf = rand(rows(pts), rows(para.samples));

	% YOUR CODE GOES HERE
	
	% for each class
	for clid = 1:rows(para.labels)
		% you know number of samples in this class so you can allocate 
		% intermediate matrix (it contains columns f1 f2 ... fn from diagram in instruction)
		onedpdfs = zeros(rows(para.samples{clid}), columns(para.samples{clid}));
		% don't forget to adjust Parzen window width
		hn = para.parzenw / sqrt(rows(para.samples{clid}))
		
		% for each sample in pts
		for ptid = 1:rows(pts)
			% for each feature
			for ftid = 1:columns(pts)  
				% fill proper column in onedpdfs with call to normpdf
				onedpdfs(:, ftid) = normpdf(pts(ptid, ftid), para.samples{clid}(:, ftid), hn);
			end
			% aggregate onedpdfs into a scalar value
			multivariatePdfshares = prod(onedpdfs, 2);
			% and store it in proper element of pdf
			pdf(ptid, clid) = mean(multivariatePdfshares);
		end
    end
end
