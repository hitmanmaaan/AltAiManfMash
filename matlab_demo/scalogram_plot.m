function scalogram_plot(wt, fw, fsx, range_db)
% SCALOGRAM_PLOT  |CWT| in dB on a log frequency axis (cwt returns log-spaced frequencies, so
%   imagesc, which assumes equal spacing, would put the rows in the wrong place).
    if nargin < 4, range_db = 60; end
    S = 20 * log10(abs(wt) + eps);
    step = max(1, floor(size(S, 2) / 2000));          % thin the time axis for plotting
    t = (0:size(S, 2) - 1) / fsx;
    pcolor(t(1:step:end), fw, S(:, 1:step:end)); shading flat;
    set(gca, 'YScale', 'log'); colorbar;
    c = clim; clim([c(2) - range_db, c(2)]);
    xlabel('Время, с'); ylabel('Частота, Гц');
end
