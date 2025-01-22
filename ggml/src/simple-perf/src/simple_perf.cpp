#include "simple_perf.h"
#include <cassert>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <sstream>
#include <fstream>

namespace {
std::string tid2str(const std::thread::id& id) {
    std::stringstream ss;
    ss << id;
    return ss.str();
}

int64_t simple_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((int64_t)ts.tv_sec*1000000) + ((int64_t)ts.tv_nsec/1000);
}

Tracer tracer{false, 0, 0};

using TraceLine = std::pair<std::string, int64_t>;

std::unordered_map<std::string, std::vector<TraceLine>> traces;
int64_t cot = 0;

}


#ifdef __cplusplus
extern "C"
{
#endif

void start_tracing_session() {
    assert(!tracer.enable);
    tracer.enable = true;
    tracer.start_time = simple_time_us();
}

void stop_tracing_session(const char *output_path) {
    tracer.enable = false;
    tracer.end_time = simple_time_us();

    std::ofstream outfile(output_path, std::ios::out);
    if (!outfile.is_open()) {
        perror("Failed to open trace file\n");
        return;
    }

    outfile << "start_time: " << tracer.start_time << "\n";
    for (auto &[k, v]: traces) {
        for (auto &t: v) {
            outfile << "thr: " << k << ", name: " << t.first << ", time: " << t.second << "\n";
        }
    }
    outfile << "end_time: " << tracer.end_time << "\n";

    outfile.close();
    printf("\nRecord trace into %s\n", output_path);
}

void begin_trace(const char *name, char *out_name_buffer) {
    auto thr = tid2str(std::this_thread::get_id());
    std::string new_name = std::string(name) + "_" +  std::to_string(cot);
    if (tracer.enable) {
        if (!traces.contains(thr)) {
            traces[thr] = {};
        }
        traces[thr].push_back({new_name, simple_time_us()});
        cot += 1;
    }
    strncpy(out_name_buffer, new_name.c_str(), new_name.size());
    out_name_buffer[new_name.size()] = '\0';
}

void end_trace(const char *name) {
    if (tracer.enable) {
        auto thr = tid2str(std::this_thread::get_id());
        assert(traces.contains(thr));
        traces[thr].push_back({"end_" + std::string(name), simple_time_us()});
    }
}


#ifdef __cplusplus
}
#endif
