#include "charon_io.h"
#include <fstream>
#include <iostream>

namespace PredictiveCore {

bool CharonAsyncIO::write_async(const std::string& filepath, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(mtx);
    
    // Launch an asynchronous file writer task
    active_writes[filepath] = std::async(std::launch::async, [filepath, data]() {
        // Direct storage bypass: open in binary mode
        std::ofstream outfile(filepath, std::ios::binary | std::ios::out);
        if (!outfile.is_open()) {
            return false;
        }
        
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size());
        outfile.close();
        return true;
    });
    
    return true;
}

void CharonAsyncIO::read_async_start(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mtx);
    
    active_reads[filepath] = std::async(std::launch::async, [filepath]() {
        std::ifstream infile(filepath, std::ios::binary | std::ios::in);
        if (!infile.is_open()) {
            return std::vector<uint8_t>();
        }
        
        // Read file size
        infile.seekg(0, std::ios::end);
        std::streamsize size = infile.tellg();
        infile.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> buffer(size);
        if (infile.read(reinterpret_cast<char*>(buffer.data()), size)) {
            return buffer;
        }
        
        return std::vector<uint8_t>();
    });
}

bool CharonAsyncIO::is_write_complete(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = active_writes.find(filepath);
    if (it == active_writes.end()) {
        return true; // No write active
    }
    
    // Check if future has completed (wait_for 0ms)
    auto status = it->second.wait_for(std::chrono::milliseconds(0));
    return status == std::future_status::ready;
}

bool CharonAsyncIO::is_read_complete(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = active_reads.find(filepath);
    if (it == active_reads.end()) {
        return true; // No read active
    }
    
    auto status = it->second.wait_for(std::chrono::milliseconds(0));
    return status == std::future_status::ready;
}

std::vector<uint8_t> CharonAsyncIO::get_read_data(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = active_reads.find(filepath);
    if (it == active_reads.end()) {
        return std::vector<uint8_t>();
    }
    
    // Blocks if not yet complete (standard future behavior)
    std::vector<uint8_t> result = it->second.get();
    active_reads.erase(it);
    return result;
}

} // namespace PredictiveCore
