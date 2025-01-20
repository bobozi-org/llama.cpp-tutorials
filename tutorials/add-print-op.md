# How To Add An Operator (Take the `cpu print operator` as an example)
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

## Add a print operator
- Sometime we need to inspect some tensors data internally, we can add a print operator for debugging.

- The workflow of llama.cpp is: creating a computing graph (`llama_build_graph`) and executing operators based on the graph (`llama_graph_compute`).

- In this tutorial, we focus on the decoding flow of llama 3.2 on the cpu side, the key functions are `build_llama` and `ggml_graph_compute`.

### Add a ggml print operator
- First, we add a simple print operator: `ggml_compute_forward_print', we can base on other operators like `ggml_compute_forward_add'.

- Add `ggml_op` definition and `GGML_API` in `ggml/include/ggml.h`
```c
    enum ggml_op {
        // ...
        GGML_OP_OPT_STEP_ADAMW,
        GGML_OP_PRINT, // 🆕: Add ggml_op type

        GGML_OP_COUNT,
    };

    // ...
    GGML_API struct ggml_tensor * ggml_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_tensor_print( // 🆕: Add GGML_API
            struct ggml_context * ctx,
            struct ggml_tensor  * a);
    // ...
```

- Modify `GGML_OP_COUNT` assertion and `GGML_OP_SYMBOL` in `ggml/src/ggml.c`
```c
// ...
// static_assert(GGML_OP_COUNT == 83, "GGML_OP_COUNT != 83");
static_assert(GGML_OP_COUNT == 84, "GGML_OP_COUNT != 84"); // 🆕: Modify GGML_OP_COUNT

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",
    // ...
    "adamw(x)",
    "print(x)", // 🆕: add op symbol
};

// static_assert(GGML_OP_COUNT == 83, "GGML_OP_COUNT != 83");
static_assert(GGML_OP_COUNT == 84, "GGML_OP_COUNT != 84"); // 🆕: Modify GGML_OP_COUNT
// ...
```

- Implement GGML_API for external call in `ggml/src/ggml.c`
```c
// ...
// ggml_tensor_print
// 🆕
static struct ggml_tensor * ggml_tensor_print_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a) {
    struct ggml_tensor *result = ggml_view_tensor(ctx, a);
    result->op = GGML_OP_PRINT;
    result->src[0] = a;
    return result;
}

struct ggml_tensor * ggml_tensor_print(
    struct ggml_context * ctx,
    struct ggml_tensor  * a) {
    return ggml_tensor_print_impl(ctx, a);
}

struct ggml_tensor * ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_add_impl(ctx, a, b, true);
}
// ...
```

- Implement the internal logic of the ggml print operator in `ggml/src/ggml-cpu/ggml-cpu.c`.
```c
// ...
// ggml_compute_forward_print
// 🆕: Add print implement
static void ggml_compute_forward_print(
     const struct ggml_compute_params * params,
    struct ggml_tensor * dst)
{
    const struct ggml_tensor * src0 = dst->src[0];
    const int ith = params->ith; // current thread
    // const int nth = params->nth;
    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(src0->type == GGML_TYPE_F32); // only support fp32 type
    if (ith == 0) { // ensure only one thread can print info
        printf("\n%s: {%ld, %ld, %ld, %ld}\n", src0->name, ne03, ne02, ne01, ne00);
        for (int i3 = 0; i3 < ne03; i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    for (int i0 = 0; i0 < ne00; i0++) {
                        float * ptr = (float *) ((char *) src0->data + (i3*nb03) + (i2*nb02) + (i1*nb01) + (i0*nb00));
                        // printf("[%d][%d][%d][%d] = %.6f\n", i3, i2, i1, i0, (double)*ptr);
                        printf("%.6f\n", (double)*ptr);
                    }
                }
            }
        }
    }
}

// ggml_compute_forward_add
// ...
```

- Modify `ggml_compute_forward` to execute print procedure in `ggml/src/ggml-cpu/ggml-cpu.c`.
```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    // ...
    switch (tensor->op) {
        case GGML_OP_PRINT: // 🆕
            {
                ggml_compute_forward_print(params, tensor);
            } break;
        // ...
```

- Modify `ggml_get_n_tasks` for compute planning in `ggml/src/ggml-cpu/ggml-cpu.c`.
```c
static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    // ...
    switch (node->op) {
        case GGML_OP_PRINT: // 🆕
            {
                n_tasks = 1;
            } break;
        // ...
```

### Call ggml print operator in graph building
- Now, we can use `ggml_tensor_print` to add a print node when building computing graph and execute it when executing the graph.
- Call ggml print operator in `src/llama.cpp`.
- NOTE: The return tensor must be assigned to the raw tensor, otherwise it will not be executed.
- NOTE: We can add some conditions (like layer id or batch size) to print specific tensors.
```cpp
    struct ggml_cgraph * build_llama() {
        // ...
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);
        inpL = ggml_tensor_print(ctx0, inpL); // 🆕
        // ... 
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
            cb(cur, "attn_norm", il);
            if (il == 0 && n_tokens == 1) {
                // only print when layer id == 0 and batch size == 1
                cur = ggml_tensor_print(ctx0, cur);
            }
            // ...
```

### (Advance) Add some additional parameters
- Sometime, we need 
