function [fp, ap] = peak_near(f, amp, f0, tol)
% PEAK_NEAR  Strongest peak within f0 ± tol, frequency refined by parabolic interpolation.
    idx = find(abs(f - f0) <= tol);
    [~, k] = max(amp(idx));
    k = idx(k);
    a = log(amp(k - 1)); b = log(amp(k)); c = log(amp(k + 1));
    fp = f(k) + 0.5 * (a - c) / (a - 2*b + c) * (f(2) - f(1));
    ap = amp(k);
end
