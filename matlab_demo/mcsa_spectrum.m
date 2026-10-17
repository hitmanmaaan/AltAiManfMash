function [f, db, amp] = mcsa_spectrum(x, fs, seconds)
% MCSA_SPECTRUM  Hann-windowed amplitude spectrum; db is relative to the strongest peak (the supply).
%   Resolution is 1/seconds Hz (60 s -> 0.0167 Hz), needed to separate close sidebands.
    n = min(numel(x), round(seconds * fs));
    seg = x(1:n) - mean(x(1:n));
    w = hann(n);
    A = abs(fft(seg .* w));
    amp = A(1:floor(n/2) + 1) * 2 / sum(w);
    f = (0:floor(n/2))' * fs / n;
    db = 20 * log10(amp / max(amp) + 1e-15);
end
