#pragma once
#include <string>
#include <vector>
#include <memory>

namespace PredictiveCore {

enum TrafficRegime {
    HUMAN_INTERACTIVE = 0,
    AGENT_CONCURRENT = 1,
    DORMANT = 2
};

enum StorageTier {
    HOT_VRAM = 0,
    WARM_DRAM = 1,
    COLD_SSD = 2
};

struct RequestSignal {
    std::string session_id;
    double timestamp_ms;
    size_t payload_size;
};

struct AllocationDecision {
    std::string session_id;
    TrafficRegime recommended_regime;
    StorageTier recommended_tier;
    bool should_evict;
};

class IEngine {
public:
    virtual ~IEngine() = default;
    virtual void notify_request(const RequestSignal& signal) = 0;
    virtual TrafficRegime evaluate_session(const std::string& session_id) = 0;
    virtual AllocationDecision get_optimal_eviction_target() = 0;
};

struct SessionMeta {
    std::string session_id;
    double last_seen_ms;
    size_t request_count;
};

class PredictiveEngine : public IEngine {
private:
    std::vector<SessionMeta> sessions;
public:
    PredictiveEngine() = default;
    ~PredictiveEngine() override = default;

    void notify_request(const RequestSignal& signal) override;
    TrafficRegime evaluate_session(const std::string& session_id) override;
    AllocationDecision get_optimal_eviction_target() override;
};

} // namespace PredictiveCore
