function T = trial_table()
% TRIAL_TABLE  Per-trial supply frequency, real shaft speed, slip and operating mode (cached).
%   Shaft speed f_r1 is measured from the f_s ± f_r1 sidebands of the current spectrum.
    cache = fullfile(fileparts(mfilename('fullpath')), 'cache', 'trial_table.mat');
    if isfile(cache)
        load(cache, 'T');
        return
    end
    fs = 25600;
    n = 45;
    [f_s, f_r1, slip, rms_R, seconds] = deal(zeros(n, 1));
    for t = 1:n
        x = load_trial(t);
        [f, ~, amp] = mcsa_spectrum(x, fs, 60);
        f_s(t) = peak_near(f, amp, 48.6, 0.6);
        lo = peak_near(f, amp, f_s(t) - 24.2, 0.4);
        hi = peak_near(f, amp, f_s(t) + 24.2, 0.4);
        f_r1(t) = ((hi - f_s(t)) + (f_s(t) - lo)) / 2;
        slip(t) = 1 - f_r1(t) / (f_s(t) / 2);  % 4-pole motor: synchronous speed = f_s/2
        rms_R(t) = std(x);
        seconds(t) = numel(x) / fs;
        fprintf('испытание %2d: f_s = %.3f Гц, вал = %.3f Гц (%.0f об/мин)\n', t, f_s(t), f_r1(t), f_r1(t)*60);
    end
    load_level = repmat("лёгкая", n, 1);
    load_level(slip > 0.0027) = "средняя";
    load_level(slip > 0.0055) = "тяжёлая";
    supply = repmat("номинал", n, 1);
    supply(f_s > 48.5) = "повышенное";
    mode = load_level + " / " + supply;
    trial = (1:n)';
    rpm = f_r1 * 60;
    T = table(trial, seconds, f_s, f_r1, rpm, slip, rms_R, load_level, supply, mode);
    save(cache, 'T');
end
