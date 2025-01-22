# How To Trace Performance Based On Perfetto
- Platform: Ubuntu 22.04 x86_64
- Model: Llama3.2-1B-Instruct Q8_0
- Toolchains: 
    - g++ 11.4.0
    - cmake 3.26.4

## Build llama.cpp
```shell
cmake -B build -S ./ -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run models for text generation
```shell
./build/bin/llama-cli -m Llama-3.2-1B-Instruct/q8_0.gguf -n 128 -p "Which is bigger, 3.11 or 3.9?" -no-cnv
```

## Trace llama.cpp's performance
- Sometime we want to know the performance of llama.cpp, so we need to trace it. I once use [perfetto](https://github.com/google/perfetto) in llama.cpp, but I found I cannot trace any infomartion, so I write a simple tool to realize the same ability with perfetto.

### Build simple perf tools in ggml
- See details in `ggml/src/ggml-cpu/simple-perf`, Note: this tool may failed when using multi-thread.

### Insert trace point in code
- Use `start_tracing_session` and `end_tracing_session` to control tracing section in `examples/main/main.cpp`.
```cpp
int main(int argc, char ** argv) {
    start_tracing_session();
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // ...
    }
    stop_tracing_session("./trace.txt");
}
```

- Use `begin_trace` and `end_trace` to record `llama_graph_compute` execution time in `src/llama.cpp`, we need to use a buffer to receive trace point name and use it when end trace.
```cpp
static int llama_decode_impl(
         llama_context & lctx,
           llama_batch   inp_batch) {
        // ...
        char name_buffer[4096];
        begin_trace("build graph", name_buffer);
        const auto compute_status = llama_graph_compute(lctx, gf, n_threads, threadpool);
        end_trace(name_buffer);
        // ...
}
```

- Use `begin_trace` and `end_trace` to record `ggml_compute_forward_mul_mat` execution time in `ggml/src/ggml-cpu/ggml-cpu.c`.
```c

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    // ...
    switch (tensor->op) {
        // ...
        case GGML_OP_MUL_MAT:
            {
                char name_buffer[4096];
                char inputname[1024] = "matmul_";
                strncat(inputname, tensor->name, strlen(tensor->name)); 
                begin_trace(inputname, name_buffer);
                ggml_compute_forward_mul_mat(params, tensor);
                end_trace(name_buffer);
            } break;
    }
    // ...
}
```

- Run llama.cpp and it will create a trace file `./trace.txt`.
```shell
./build/bin/llama-cli -m /data1/models/Llama-3.2-1B-Instruct/q8_0.gguf -n 12 -p "Which is bigger, 3.11 or 3.9?" -no-cnv
...
Record trace into ./trace.txt
...
```

- Use script to convert `./trace.txt` to json `./out_trace.json` that [perfetto.ui](https://ui.perfetto.dev) can display.
```shell
python ./tutorials/trace_converter.py -f ./trace.txt -o ./out_trace.json -t ./tutorials/trace_template.json
```

- Then use [perfetto.ui](https://ui.perfetto.dev) to view and analyze the trace. (You can use `demo-out_trace.json` which generated from `demo-trace.txt`)
