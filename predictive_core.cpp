#include "predictive_core.h"
#include <algorithm>
#include <iostream>

namespace PredictiveCore {

void PredictiveEngine::notify_request(const RequestSignal& signal) {
    auto it = std::find_if(sessions.begin(), sessions.end(), [&](const SessionMeta& meta) {
        return meta.session_id == signal.session_id;
    });

    if (it != sessions.end()) {
        it->last_seen_ms = signal.timestamp_ms;
        it->request_count++;
    } else {
        SessionMeta meta;
        meta.session_id = signal.session_id;
        meta.last_seen_ms = signal.timestamp_ms;
        meta.request_count = 1;
        sessions.push_back(meta);
    }
}

TrafficRegime PredictiveEngine::evaluate_session(const std::string& session_id) {
    auto it = std::find_if(sessions.begin(), sessions.end(), [&](const SessionMeta& meta) {
        return meta.session_id == session_id;
    });

    // Default current time mockup is based on last seen of all sessions
    double current_time = 0.0;
    for (const auto& meta : sessions) {
        if (meta.last_seen_ms > current_time) {
            current_time = meta.last_seen_ms;
        }
    }

    if (it != sessions.end()) {
        // Dormancy threshold: > 5 minutes (300,000 ms) in simulated time
        if (current_time - it->last_seen_ms > 300000.0) {
            return DORMANT;
        }
    }

    if (session_id.find("agent") != std::string::npos || session_id.find("bot") != std::string::npos) {
        return AGENT_CONCURRENT;
    }
    return HUMAN_INTERACTIVE;
}

AllocationDecision PredictiveEngine::get_optimal_eviction_target() {
    AllocationDecision decision;
    decision.should_evict = false;
    decision.session_id = "";
    decision.recommended_regime = HUMAN_INTERACTIVE;
    decision.recommended_tier = WARM_DRAM;

    // 1. Prioritize Cold SSD eviction of Dormant sessions
    for (const auto& meta : sessions) {
        if (evaluate_session(meta.session_id) == DORMANT) {
            decision.session_id = meta.session_id;
            decision.should_evict = true;
            decision.recommended_regime = DORMANT;
            decision.recommended_tier = COLD_SSD;
            return decision;
        }
    }

    // 2. Fall back to Warm DRAM eviction of Human Interactive sessions
    for (const auto& meta : sessions) {
        if (evaluate_session(meta.session_id) == HUMAN_INTERACTIVE) {
            decision.session_id = meta.session_id;
            decision.should_evict = true;
            decision.recommended_regime = HUMAN_INTERACTIVE;
            decision.recommended_tier = WARM_DRAM;
            return decision;
        }
    }

    return decision;
}

} // namespace PredictiveCore
