function [band, snr_db, fe, ae] = best_band(x, fs, lines, fmin)
% BEST_BAND  Targeted band selection ("protrugram"): which high-frequency band has the strongest
%   envelope lines at the expected defect frequencies (vector lines, Гц)?
%   |STFT|^2 of every 200 Hz bin is the squared envelope of that band (sampled at fs/16); adjacent
%   bins are summed into 400/800/1600 Hz bands. Same algorithm as analysis/envelope.py: env_sel.
%   Returns the best band [lo hi], its line SNR (dB over local median) and its envelope spectrum.
    if nargin < 4, fmin = 1000; end
    nper = 128; hop = 16;
    [S, f] = stft(x(:) - mean(x), fs, 'Window', hann(nper, 'periodic'), 'OverlapLength', nper - hop, ...
                  'FrequencyRange', 'onesided');
    P = abs(S).^2;
    fenv = fs / hop;
    df = f(2) - f(1);
    first = find(f >= fmin, 1);
    snr_db = -inf;
    for w = [1 2 4 8]
        for i = first:max(1, floor(w/2)):numel(f) - w
            e = sum(P(i:i+w-1, :), 1)';
            e = e / mean(e) - 1;
            n = numel(e); win = hann(n);
            A = abs(fft(e .* win)); a = A(1:floor(n/2)+1) * 2 / sum(win);
            fa = (0:floor(n/2))' * fenv / n;
            v = line_snr(fa, a, lines);
            if v > snr_db
                snr_db = v; band = [f(i) - df/2, f(i+w-1) + df/2]; fe = fa; ae = a;
            end
        end
    end
end

function v = line_snr(f, a, lines)
    r = [];
    for d = lines(:)'
        tol = max(0.006 * d, 0.15);
        win = abs(f - d) <= tol;
        ref = abs(f - d) <= 3 & abs(f - d) > max(tol, 0.5);
        if any(win) && nnz(ref) >= 10
            r(end+1) = max(a(win).^2) / median(a(ref).^2); %#ok<AGROW>
        end
    end
    v = 10 * log10(mean(r));
end
