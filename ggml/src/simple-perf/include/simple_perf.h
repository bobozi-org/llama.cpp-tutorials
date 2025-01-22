#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

struct Tracer {
    bool enable;
    int64_t start_time;
    int64_t end_time;
};

void start_tracing_session(void);
void stop_tracing_session(const char *output_path);

void begin_trace(const char *name, char *out_name_buffer);
void end_trace(const char *name);

#ifdef __cplusplus
}
#endif
