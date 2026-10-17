% DEMO_KNP_MATLAB  End-to-end demo: generate a stimulus in MATLAB, run it
% through the Kaspersky Neuromorphic Platform (spiking network on the CPU
% backend, running inside Docker), and visualize input vs. output spikes.

input_steps = [0 5 10 15];
num_steps = 20;

[output_steps, input_steps] = knp_run_network(input_steps, num_steps, 'cpu');

fprintf('Input spike steps:  %s\n', mat2str(input_steps));
fprintf('Output spike steps: %s\n', mat2str(output_steps(:)'));

fig = figure('Visible', 'off');
hold on;
stem(input_steps, ones(size(input_steps)), 'b', 'DisplayName', 'Input spikes');
stem(output_steps, 0.5*ones(size(output_steps)), 'r', 'DisplayName', 'Output spikes');
xlim([0 num_steps-1]);
ylim([0 1.2]);
xlabel('Simulation step');
yticks([]);
title('KNP spiking network: input vs. output (CPU backend)');
legend('Location', 'northoutside', 'Orientation', 'horizontal');
grid on;
box on;
saveas(fig, fullfile(fileparts(mfilename('fullpath')), 'spike_raster.png'));
fprintf('Saved plot to spike_raster.png\n');
