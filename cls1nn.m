function labels = cls1nn(x, ts)
    if isvector(x)
        x = x(:).';
    end
    ytr = ts(:,1);
    Xtr = ts(:,2:end);
    D = pdist2(x, Xtr, 'euclidean');
    [~, idx] = min(D, [], 2);
    labels = ytr(idx);
end

