function pdf = normpdf(x, mu, sig)
  % computes 1D normal distribution for x: ~N(mu,sig)
  pdf = exp(-(x - mu).^2/(2 * sig.^2)) / sqrt(2*pi*sig.^2);
end
