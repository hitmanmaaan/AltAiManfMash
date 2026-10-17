/**
 * @file main.cpp
 * @brief Minimal spiking network executed on the AltAI backend (software
 * "golden model" emulator of the AltAI neuromorphic chip).
 * @details Based on examples/simple-network, adapted to the AltAILIF neuron
 * type supported by the AltAI backend.
 */

#include <knp/framework/io/out_converters/convert_set.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/synapse-traits/delta.h>

#include <filesystem>
#include <iostream>


using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using AltaiPopulation = knp::core::Population<knp::neuron_traits::AltAILIF>;

inline std::optional<DeltaProjection::Synapse> input_projection_gen(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 1, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

inline std::optional<DeltaProjection::Synapse> synapse_generator(size_t /*index*/)  // NOLINT
{
    return DeltaProjection::Synapse{{1.0, 6, knp::synapse_traits::OutputType::EXCITATORY}, 0, 0};
}

inline knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF> neuron_generator(size_t)  // NOLINT
{
    return knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
}


int main(int argc, const char *const argv[])
{
    AltaiPopulation population{neuron_generator, 1};
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

    auto input_gen = [](knp::core::Step step) -> knp::core::messaging::SpikeData
    {
        if (step % 5 == 0)
        {
            knp::core::messaging::SpikeData s;
            s.push_back(0);
            return s;
        }
        return {};
    };

    // AltAI backend shared library, installed system-wide from the
    // knp-altai-backend .deb package (not next to this binary).
    std::filesystem::path backend_path =
        (argc > 1) ? std::filesystem::path(argv[1]) : std::filesystem::path("/usr/lib/knp-altai-backend");

    std::cout << "Loading backend from: " << backend_path.string() << std::endl;

    knp::framework::BackendLoader backend_loader;
    auto backend = backend_loader.load(backend_path);

    auto devices = backend->get_devices();
    std::cout << "Devices reported by backend: " << devices.size() << std::endl;
    for (const auto &d : devices) std::cout << "  - " << d->get_name() << std::endl;
    if (!devices.empty())
    {
        backend->select_device(std::move(devices.front()));
        std::cout << "Device selected." << std::endl;
    }

    knp::framework::ModelExecutor model_executor(model, backend, {{i_channel_uid, input_gen}});

    std::cout << "Model loaded onto backend OK." << std::endl;

    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.start([](size_t step) { return step < 20; });

    std::cout << "Run finished OK." << std::endl;

    std::vector<knp::core::Step> results;
    const auto &spikes = out_channel.update();
    results.reserve(spikes.size());

    std::transform(
        spikes.cbegin(), spikes.cend(), std::back_inserter(results),
        [](const auto &spike_msg) { return spike_msg.header_.send_time_; });

    std::cout << "Spikes at steps: ";
    for (const auto &s : results) std::cout << s << " ";
    std::cout << std::endl;
}
