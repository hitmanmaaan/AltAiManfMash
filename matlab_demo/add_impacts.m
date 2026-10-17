function y = add_impacts(x, fs, rate, level_db, rng_seed)
% ADD_IMPACTS  Synthetic spall: decaying bursts at a random 2-6 kHz resonance, one per 1/rate s with
%   1 % random slip, added to x. level_db = burst peak relative to the RMS of x in 1-10 kHz.
%   Same model as analysis/synth/common.py: inject_impacts (outer race variant).
    rng(rng_seed);
    x = x(:);
    n = numel(x);
    T = 1 / rate;
    tk = cumsum(T * (1 + 0.01 * randn(ceil(n / fs / T) + 10, 1))) + rand * T;
    tk = tk(tk < n / fs - 0.01);
    f_res = 2000 + 4000 * rand;
    tau = 0.4e-3;
    th = (0:round(8 * tau * fs) - 1)' / fs;
    h = exp(-th / tau) .* sin(2 * pi * f_res * th);
    s = zeros(n, 1);
    s(round(tk * fs) + 1) = 1;
    s = filter(h, 1, s);
    [b, a] = butter(4, [1000 10000] / (fs/2), 'bandpass');
    sigma = std(filtfilt(b, a, x - mean(x)));
    y = x + s * 10^(level_db / 20) * sigma / max(abs(h));
end
