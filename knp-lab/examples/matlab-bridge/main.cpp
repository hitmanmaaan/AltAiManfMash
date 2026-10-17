/**
 * @file main.cpp
 * @brief File-based bridge between MATLAB and the Kaspersky Neuromorphic
 * Platform: reads an input spike train from a CSV file, runs it through a
 * spiking network on the requested KNP backend, and writes the resulting
 * output spike train to another CSV file.
 * @details Usage:
 *   matlab-bridge <backend_path> <input_csv> <output_csv> <num_steps>
 * input_csv:  one integer step number per line at which an input spike
 *             is injected.
 * output_csv: one integer step number per line at which the output
 *             neuron spiked (written by this program).
 */

#include <knp/framework/io/out_converters/convert_set.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/blifat.h>
#include <knp/synapse-traits/delta.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>


using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using BLIFATPopulation = knp::core::Population<knp::neuron_traits::BLIFATNeuron>;

inline std::optional<DeltaProjection::Synapse> input_projection_gen(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

inline std::optional<DeltaProjection::Synapse> synapse_generator(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 6, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> neuron_generator(size_t)  // NOLINT
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>{};
}


std::set<knp::core::Step> read_steps_csv(const std::filesystem::path &path)
{
    std::set<knp::core::Step> steps;
    std::ifstream in(path);
    if (!in) return steps;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        steps.insert(static_cast<knp::core::Step>(std::stoul(line)));
    }
    return steps;
}


void write_steps_csv(const std::filesystem::path &path, const std::vector<knp::core::Step> &steps)
{
    std::ofstream out(path);
    for (const auto &s : steps) out << s << "\n";
}


int main(int argc, const char *const argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <backend_path> <input_csv> <output_csv> <num_steps>" << std::endl;
        return 1;
    }

    std::filesystem::path backend_path = argv[1];
    std::filesystem::path input_csv = argv[2];
    std::filesystem::path output_csv = argv[3];
    const size_t num_steps = std::stoul(argv[4]);

    const std::set<knp::core::Step> input_steps = read_steps_csv(input_csv);
    std::cout << "Read " << input_steps.size() << " input spike steps from " << input_csv << std::endl;

    BLIFATPopulation population{neuron_generator, 1};
    DeltaProjection loop_projection = DeltaProjection{population.get_uid(), population.get_uid(), synapse_generator, 1};
    DeltaProjection input_projection =
        DeltaProjection{knp::core::UID{false}, population.get_uid(), input_projection_gen, 1};

    knp::core::UID input_uid = input_projection.get_uid();
    knp::core::UID output_uid = population.get_uid();

    knp::framework::Network network;
    network.add_population(std::move(population));
    network.add_projection(std::move(input_projection));
    network.add_projection(std::move(loop_projection));

    knp::core::UID i_channel_uid, o_channel_uid;

    knp::framework::Model model(std::move(network));
    model.add_input_channel(i_channel_uid, input_uid);
    model.add_output_channel(o_channel_uid, output_uid);

    auto input_gen = [&input_steps](knp::core::Step step) -> knp::core::messaging::SpikeData
    {
        if (input_steps.count(step))
        {
            knp::core::messaging::SpikeData s;
            s.push_back(0);
            return s;
        }
        return {};
    };

    std::cout << "Loading backend from: " << backend_path.string() << std::endl;

    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(backend_path);

    knp::framework::ModelExecutor model_executor(model, backend, {{i_channel_uid, input_gen}});

    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.start([num_steps](size_t step) { return step < num_steps; });

    std::vector<knp::core::Step> results;
    const auto &spikes = out_channel.update();
    results.reserve(spikes.size());
    std::transform(
        spikes.cbegin(), spikes.cend(), std::back_inserter(results),
        [](const auto &spike_msg) { return spike_msg.header_.send_time_; });

    write_steps_csv(output_csv, results);
    std::cout << "Wrote " << results.size() << " output spike steps to " << output_csv << std::endl;
}
