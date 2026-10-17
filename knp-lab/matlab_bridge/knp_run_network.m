function [output_steps, input_steps] = knp_run_network(input_steps, num_steps, backend)
% KNP_RUN_NETWORK  Run a spiking network on the Kaspersky Neuromorphic
% Platform (KNP) through a file-based bridge and Docker, and return the
% output spike steps.
%
%   [output_steps, input_steps] = knp_run_network(input_steps, num_steps, backend)
%
%   input_steps : vector of simulation step indices at which an input
%                 spike is injected into the network (default: [0 5 10 15]).
%   num_steps   : number of simulation steps to run (default: 20).
%   backend     : 'cpu' (default, always available) or 'altai' (golden
%                 model emulator of the AltAI neuromorphic chip -- not
%                 currently usable, see README).
%
% This function shells out to `docker exec` to run the compiled KNP
% "matlab-bridge" C++ example inside the knp-run container. Input/output
% are exchanged as CSV files through the Docker bind mount that maps this
% directory to /work/matlab_bridge inside the container.

    if nargin < 1 || isempty(input_steps)
        input_steps = [0 5 10 15];
    end
    if nargin < 2 || isempty(num_steps)
        num_steps = 20;
    end
    if nargin < 3 || isempty(backend)
        backend = 'cpu';
    end

    here = fileparts(mfilename('fullpath'));
    input_csv = fullfile(here, 'input_spikes.csv');
    output_csv = fullfile(here, 'output_spikes.csv');

    % Clean any previous output so a failed run can't be mistaken for success.
    if exist(output_csv, 'file')
        delete(output_csv);
    end

    writematrix(input_steps(:), input_csv);

    switch backend
        case 'cpu'
            backend_path = './knp-cpu-single-threaded-backend';
        case 'altai'
            backend_path = '/usr/lib/knp-altai-backend';
        otherwise
            error('knp_run_network:backend', 'Unknown backend "%s" (use ''cpu'' or ''altai'').', backend);
    end

    docker_bin = '/usr/local/bin/docker';
    cmd = sprintf(['%s exec knp-run bash -c "cd /work/repo/build-gcc14/bin && ' ...
                   './matlab-bridge %s /work/matlab_bridge/input_spikes.csv ' ...
                   '/work/matlab_bridge/output_spikes.csv %d"'], ...
                   docker_bin, backend_path, num_steps);

    [status, cmdout] = system(cmd);
    disp(cmdout);

    if status ~= 0
        error('knp_run_network:run', 'KNP execution failed (exit %d). See output above.', status);
    end

    if ~exist(output_csv, 'file')
        error('knp_run_network:noOutput', 'KNP ran but produced no output file.');
    end

    output_steps = readmatrix(output_csv);
end
