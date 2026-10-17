function x = load_trial(trial, phase)
% LOAD_TRIAL  Stator current of one trial (amperes, 25.6 kHz). phase: 'R' (default), 'S' or 'T'.
    if nargin < 2
        phase = 'R';
    end
    root = fileparts(fileparts(mfilename('fullpath')));
    s = load(fullfile(root, 'data', 'mat', sprintf('current_%d.mat', trial)), phase);
    x = double(s.(phase)(:));
end
