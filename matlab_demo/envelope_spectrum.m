function [f, amp, e, fse] = envelope_spectrum(x, fs, band, squared)
% ENVELOPE_SPECTRUM  Spectrum of the (squared) envelope of x in a frequency band.
%   band = [lo hi] Гц. Envelope = |hilbert(bandpass(x))| (squared if squared = true), normalised to
%   its mean minus 1, so a line of amplitude 0.01 means 1 % modulation. For bands below 1 kHz the
%   signal is first decimated to 1024 Hz. The envelope itself is resampled to 1024 Hz.
    if nargin < 4, squared = false; end
    x = x(:) - mean(x);
    fse = 1024;
    if band(2) <= 500
        x = resample(x, fse, fs);           % low band: work at 1024 Hz directly
        fsx = fse;
    else
        fsx = fs;
    end
    [b, a] = butter(4, band / (fsx/2), 'bandpass');
    y = filtfilt(b, a, x);
    e = abs(hilbert(y));
    if squared, e = e.^2; end
    if fsx ~= fse, e = resample(e - mean(e), fse, fsx) + mean(e); end
    k = round(0.02 * numel(e));             % edge effects of filter / Hilbert
    e = e(k+1:end-k);
    e = e / mean(e) - 1;
    n = numel(e);
    w = hann(n);
    A = abs(fft(e .* w));
    amp = A(1:floor(n/2) + 1) * 2 / sum(w);
    f = (0:floor(n/2))' * fse / n;
end
