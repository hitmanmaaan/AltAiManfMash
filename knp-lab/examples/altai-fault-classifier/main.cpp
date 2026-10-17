/**
 * @file main.cpp
 * @brief One-layer ternary spiking classifier of motor faults for the AltAI neuron model.
 * @details Usage:
 *   altai-fault-classifier <backend_path> <weights.csv> <input_spikes.csv> <output_spikes.csv> [--select-device]
 *
 * weights.csv: first line "n_inputs,n_outputs"; then one line per output neuron:
 *   threshold,w_0,...,w_{n_inputs-1}  with w in {-1, 0, 1} and integer threshold >= 1.
 * input_spikes.csv: one line per sample; comma-separated indices of the active (spiking) inputs
 *   (an empty line means no active input). Sample k is presented at network step k.
 * output_spikes.csv: "step,neuron" for every output spike.
 *
 * Each output neuron is AltAILIF with do_not_save_ = true, so its potential is cleared every
 * step: it fires iff sum_i w_i * x_i >= threshold for the sample presented one step earlier.
 */

#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/synapse-traits/delta.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using AltaiPopulation = knp::core::Population<knp::neuron_traits::AltAILIF>;
using AltaiParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>;


std::vector<int> split_ints(const std::string &line)
{
    std::vector<int> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
    {
        if (!cell.empty()) out.push_back(std::stoi(cell));
    }
    return out;
}


int main(int argc, const char *const argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <backend_path> <weights.csv> <input_spikes.csv> <output_spikes.csv> [--select-device]"
                  << std::endl;
        return 1;
    }
    const std::filesystem::path backend_path = argv[1];
    const bool select_device = argc > 5 && std::string(argv[5]) == "--select-device";

    std::ifstream wf(argv[2]);
    std::string line;
    std::getline(wf, line);
    const auto dims = split_ints(line);
    const size_t n_in = dims.at(0), n_out = dims.at(1);
    std::vector<uint16_t> thresholds;
    std::vector<std::vector<int>> weights;
    while (std::getline(wf, line) && weights.size() < n_out)
    {
        auto row = split_ints(line);
        thresholds.push_back(static_cast<uint16_t>(row.at(0)));
        weights.emplace_back(row.begin() + 1, row.end());
    }

    std::vector<knp::core::messaging::SpikeData> samples;
    std::ifstream sf(argv[3]);
    while (std::getline(sf, line))
    {
        knp::core::messaging::SpikeData s;
        for (int i : split_ints(line)) s.push_back(static_cast<knp::core::messaging::SpikeIndex>(i));
        samples.push_back(s);
    }
    std::cout << "Network: " << n_in << " binary inputs -> " << n_out << " AltAILIF neurons; "
              << samples.size() << " samples" << std::endl;

    AltaiPopulation population{
        [&thresholds](size_t j)
        {
            AltaiParams p;
            p.activation_threshold_ = thresholds[j];
            p.do_not_save_ = true;
            p.is_reset_ = true;
            p.potential_leak_ = 0;
            return p;
        },
        n_out};

    std::vector<DeltaProjection::Synapse> synapses;
    for (size_t j = 0; j < n_out; ++j)
        for (size_t i = 0; i < n_in; ++i)
            if (weights[j][i] != 0)
                synapses.push_back(DeltaProjection::Synapse{
                    {static_cast<float>(weights[j][i]), 1, knp::synapse_traits::OutputType::EXCITATORY}, i, j});
    std::cout << "Non-zero ternary synapses: " << synapses.size() << std::endl;

    DeltaProjection input_projection{
        knp::core::UID{false}, population.get_uid(),
        [&synapses](size_t k) -> std::optional<DeltaProjection::Synapse> { return synapses[k]; }, synapses.size()};

    const knp::core::UID input_uid = input_projection.get_uid();
    const knp::core::UID output_uid = population.get_uid();

    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection(std::move(input_projection));

    knp::core::UID i_channel_uid, o_channel_uid;
    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_uid);
    model.add_output_channel(o_channel_uid, output_uid);

    auto input_gen = [&samples](knp::core::Step step) -> knp::core::messaging::SpikeData
    { return step < samples.size() ? samples[step] : knp::core::messaging::SpikeData{}; };

    std::cout << "Loading backend from: " << backend_path.string() << std::endl;
    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(backend_path);
    if (select_device)
    {
        auto devices = backend->get_devices();
        std::cout << "Devices reported by backend: " << devices.size() << std::endl;
        for (const auto &d : devices) std::cout << "  - " << d->get_name() << std::endl;
        if (!devices.empty()) backend->select_device(std::move(devices.front()));
    }

    knp::framework::ModelExecutor model_executor(model, backend, {{i_channel_uid, input_gen}});
    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);
    const size_t n_steps = samples.size() + 2;
    model_executor.start([n_steps](size_t step) { return step < n_steps; });

    std::ofstream of(argv[4]);
    size_t n_spikes = 0;
    for (const auto &msg : out_channel.update())
        for (auto idx : msg.neuron_indexes_)
        {
            of << msg.header_.send_time_ << "," << idx << "\n";
            ++n_spikes;
        }
    std::cout << "Output spikes: " << n_spikes << " -> " << argv[4] << std::endl;
}
