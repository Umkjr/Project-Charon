#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "predictive_core.h"
#include "charon_io.h"

namespace py = pybind11;

PYBIND11_MODULE(charon_core, m) {
    m.doc() = "Charon Predictive Memory Core Python Bindings";

    // Bind TrafficRegime Enum
    py::enum_<PredictiveCore::TrafficRegime>(m, "TrafficRegime")
        .value("HUMAN_INTERACTIVE", PredictiveCore::TrafficRegime::HUMAN_INTERACTIVE)
        .value("AGENT_CONCURRENT", PredictiveCore::TrafficRegime::AGENT_CONCURRENT)
        .value("DORMANT", PredictiveCore::TrafficRegime::DORMANT)
        .export_values();

    // Bind StorageTier Enum
    py::enum_<PredictiveCore::StorageTier>(m, "StorageTier")
        .value("HOT_VRAM", PredictiveCore::StorageTier::HOT_VRAM)
        .value("WARM_DRAM", PredictiveCore::StorageTier::WARM_DRAM)
        .value("COLD_SSD", PredictiveCore::StorageTier::COLD_SSD)
        .export_values();

    // Bind RequestSignal Struct
    py::class_<PredictiveCore::RequestSignal>(m, "RequestSignal")
        .def(py::init<>())
        .def(py::init([](const std::string& session_id, double timestamp_ms, size_t payload_size) {
            auto signal = std::make_unique<PredictiveCore::RequestSignal>();
            signal->session_id = session_id;
            signal->timestamp_ms = timestamp_ms;
            signal->payload_size = payload_size;
            return signal;
        }), py::arg("session_id") = "", py::arg("timestamp_ms") = 0.0, py::arg("payload_size") = 0)
        .def_readwrite("session_id", &PredictiveCore::RequestSignal::session_id)
        .def_readwrite("timestamp_ms", &PredictiveCore::RequestSignal::timestamp_ms)
        .def_readwrite("payload_size", &PredictiveCore::RequestSignal::payload_size);

    // Bind AllocationDecision Struct
    py::class_<PredictiveCore::AllocationDecision>(m, "AllocationDecision")
        .def(py::init<>())
        .def_readwrite("session_id", &PredictiveCore::AllocationDecision::session_id)
        .def_readwrite("recommended_regime", &PredictiveCore::AllocationDecision::recommended_regime)
        .def_readwrite("recommended_tier", &PredictiveCore::AllocationDecision::recommended_tier)
        .def_readwrite("should_evict", &PredictiveCore::AllocationDecision::should_evict);

    // Bind IEngine Interface
    py::class_<PredictiveCore::IEngine, std::shared_ptr<PredictiveCore::IEngine>>(m, "IEngine")
        .def("notify_request", &PredictiveCore::IEngine::notify_request, py::arg("signal"))
        .def("evaluate_session", &PredictiveCore::IEngine::evaluate_session, py::arg("session_id"))
        .def("get_optimal_eviction_target", &PredictiveCore::IEngine::get_optimal_eviction_target);

    // Bind PredictiveEngine Class
    py::class_<PredictiveCore::PredictiveEngine, PredictiveCore::IEngine, std::shared_ptr<PredictiveCore::PredictiveEngine>>(m, "PredictiveEngine")
        .def(py::init<>());

    // Bind CharonAsyncIO Class (Fast Direct binary mapping)
    py::class_<PredictiveCore::CharonAsyncIO>(m, "CharonAsyncIO")
        .def(py::init<>())
        .def("write_async", [](PredictiveCore::CharonAsyncIO& self, const std::string& filepath, py::bytes data) {
            std::string s = data;
            std::vector<uint8_t> buffer(s.begin(), s.end());
            return self.write_async(filepath, buffer);
        }, py::arg("filepath"), py::arg("data"))
        .def("read_async_start", &PredictiveCore::CharonAsyncIO::read_async_start, py::arg("filepath"))
        .def("is_write_complete", &PredictiveCore::CharonAsyncIO::is_write_complete, py::arg("filepath"))
        .def("is_read_complete", &PredictiveCore::CharonAsyncIO::is_read_complete, py::arg("filepath"))
        .def("get_read_data", [](PredictiveCore::CharonAsyncIO& self, const std::string& filepath) {
            std::vector<uint8_t> buffer = self.get_read_data(filepath);
            std::string s(buffer.begin(), buffer.end());
            return py::bytes(s);
        }, py::arg("filepath"));
}
