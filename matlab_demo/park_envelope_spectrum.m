function [f, amp] = park_envelope_spectrum(X, fs)
% PARK_ENVELOPE_SPECTRUM  Envelope spectrum of the Park vector |i_alpha + j i_beta| from three phases.
%   X = [R S T] (n x 3). Unlike a single phase, the Park vector of a modulated three-phase current is
%   A(1 + m g(t)) exactly, so a fault at F > f_s gives one line at F, not two (F and |F - 2 f_s|).
%   Phases are scaled to equal RMS first (phase T is measured ~23 % low).
    fse = 1024;
    Xd = resample(X - mean(X), fse, fs);
    [b, a] = butter(4, [5 400] / (fse/2), 'bandpass');
    Xd = filtfilt(b, a, Xd);
    Xd = Xd ./ std(Xd);
    e = abs((2*Xd(:,1) - Xd(:,2) - Xd(:,3)) / 3 + 1i * (Xd(:,2) - Xd(:,3)) / sqrt(3));
    k = round(0.02 * numel(e));
    e = e(k+1:end-k);
    e = e / mean(e) - 1;
    n = numel(e);
    w = hann(n);
    A = abs(fft(e .* w));
    amp = A(1:floor(n/2) + 1) * 2 / sum(w);
    f = (0:floor(n/2))' * fse / n;
end
