function e = envelope_band(x, fs)
% ENVELOPE_BAND  Relative envelope of the supply carrier (5-400 Hz band) at 1024 Hz, for scalograms.
    xd = resample(x(:) - mean(x), 1024, fs);
    [b, a] = butter(4, [5 400] / 512, 'bandpass');
    e = abs(hilbert(filtfilt(b, a, xd)));
    k = round(0.02 * numel(e));
    e = e(k+1:end-k);
    e = e / mean(e) - 1;
end
