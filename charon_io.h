#pragma once
#include <string>
#include <vector>
#include <future>
#include <unordered_map>
#include <mutex>

namespace PredictiveCore {

class CharonAsyncIO {
private:
    std::mutex mtx;
    std::unordered_map<std::string, std::future<bool>> active_writes;
    std::unordered_map<std::string, std::future<std::vector<uint8_t>>> active_reads;

public:
    CharonAsyncIO() = default;
    ~CharonAsyncIO() = default;

    // Trigger asynchronous file write to SSD
    bool write_async(const std::string& filepath, const std::vector<uint8_t>& data);

    // Trigger asynchronous file read from SSD
    void read_async_start(const std::string& filepath);
    
    // Poll to check if the async operation has completed
    bool is_write_complete(const std::string& filepath);
    bool is_read_complete(const std::string& filepath);

    // Fetch the read data (blocks until complete if called before ready)
    std::vector<uint8_t> get_read_data(const std::string& filepath);
};

} // namespace PredictiveCore
