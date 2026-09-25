"""Chapter 24 — Inside the CUDA Backend."""
from gen import *

PART = "IV"

RELU_PTX = """
    .visible .entry relu_f32(
        .param .u64 p_x,
        .param .u64 p_y,
        .param .u32 p_n
    )
    {
        ...                                   // register declarations
        mov.u32 %r1, %ctaid.x;                // block index
        mov.u32 %r2, %ntid.x;                 // threads per block (256)
        mov.u32 %r3, %tid.x;                  // thread index in the block
        mad.lo.u32 %i, %r1, %r2, %r3;         // i = block * 256 + thread
        ld.param.u32 %n, [p_n];
        setp.ge.u32 %p0, %i, %n;              // past the end? skip
        @%p0 bra DONE;
        mul.wide.u32 %off, %i, 4;             // byte offset of element i
        ld.param.u64 %b_x, [p_x];
        cvta.to.global.u64 %b_x, %b_x;
        add.u64 %a_x, %b_x, %off;
        ld.param.u64 %b_y, [p_y];
        cvta.to.global.u64 %b_y, %b_y;
        add.u64 %a_y, %b_y, %off;
        ld.global.f32 %f1, [%a_x];            // x[i]
        max.f32 %f2, %f1, 0f00000000;         // max(x[i], 0)
        st.global.f32 [%a_y], %f2;            // y[i] = ...
    DONE:
        ret;
    }
"""


def flow_svg():
    w, h = 470, 120
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    boxes = [("C# builds PTX text", 10), ("driver JIT → GPU code", 130), ("cuLaunchKernel on stream", 250), ("GPU runs 1000s of threads", 370)]
    for label, x in boxes:
        p.append(f'<rect x="{x}" y="30" width="100" height="40" rx="5" fill="#e6f2ef" stroke="#0f6b5c"/>')
        words = label.split(" ")
        mid = len(words) // 2
        p.append(svg_text(x + 50, 47, " ".join(words[:mid]), 7.8, "#0f6b5c"))
        p.append(svg_text(x + 50, 60, " ".join(words[mid:]), 7.8, "#0f6b5c"))
    for x in (110, 230, 350):
        p.append(f'<line x1="{x}" y1="50" x2="{x + 20}" y2="50" stroke="#56606a"/>')
        p.append(f'<polygon points="{x + 20},50 {x + 14},47 {x + 14},53" fill="#56606a"/>')
    p.append(svg_text(60, 90, "once per process", 7.4, "#56606a"))
    p.append(svg_text(180, 90, "once per GPU (cached", 7.4, "#56606a"))
    p.append(svg_text(180, 101, "by the driver)", 7.4, "#56606a"))
    p.append(svg_text(300, 90, "every operation", 7.4, "#56606a"))
    p.append(svg_text(420, 90, "asynchronously", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "cuda",
            "The CUDA backend runs every operation on an NVIDIA GPU using only what the display driver provides. It "
            "writes its GPU programs as PTX text in C#, has the driver compile them for the installed card, manages "
            "GPU memory with a caching allocator, queues work on its own stream and can record whole sequences of work "
            "as CUDA graphs. This chapter walks through each piece, so the GPU behaviour described in earlier "
            "chapters has no mystery left.",
            "P/Invoke into the driver API (<code>nvcuda.dll</code> / <code>libcuda.so.1</code>): no CUDA Toolkit, no cuBLAS, no cuDNN, no native code of our own.",
            "57 kernels are generated as PTX for <code>sm_50</code> and JIT-compiled by the driver for any GPU from Maxwell to Blackwell.",
            "One context and one stream per GPU; operations return after queuing; reads synchronize.",
            "Memory: a caching allocator keyed by size; cached blocks are reused, released on demand or when a limit is reached.",
            "CUDA graphs: stream capture records a step; one <code>cuGraphLaunch</code> replays it.",
        ),
        diagram("Figure 24.1 — From C# to GPU threads", flow_svg(),
                "The PTX text is built once per process; compiled code is cached by the driver on disk between runs."),
        h2("24.1 Talking to the driver"),
        reftable(["Driver call", "Used for"], [
            ["<code>cuInit</code>, <code>cuDeviceGetCount</code>, <code>cuDeviceGet</code>, <code>cuDeviceGetName</code>, <code>cuDeviceTotalMem</code>", "Detection and <code>Device.Name</code> (" + ch("devices") + ")"],
            ["<code>cuDevicePrimaryCtxRetain</code>, <code>cuCtxSetCurrent</code>", "One context per GPU, made current on whichever thread calls"],
            ["<code>cuStreamCreate</code>", "The backend's own stream: all kernels and async copies go there"],
            ["<code>cuModuleLoadDataEx</code>, <code>cuModuleGetFunction</code>", "JIT-compiling the PTX and finding each kernel"],
            ["<code>cuMemAlloc</code>, <code>cuMemFree</code>, <code>cuMemsetD32Async</code>", "Device memory"],
            ["<code>cuMemcpyHtoD</code>, <code>cuMemcpyDtoH</code>, <code>cuMemcpyDtoDAsync</code>", "Uploads, downloads (synchronizing) and device copies"],
            ["<code>cuLaunchKernel</code>", "Every operation"],
            ["<code>cuCtxSynchronize</code>", "<code>Device.Synchronize()</code>"],
            ["<code>cuStreamBeginCapture</code>, <code>cuStreamEndCapture</code>, <code>cuGraphInstantiateWithFlags</code>, <code>cuGraphLaunch</code>", "Compute graphs (" + ch("generation") + ")"],
        ], caption="Table 24.1 — The driver API calls the library makes"),
        para("The calls are declared with .NET's <code>LibraryImport</code> source generator and resolved against the "
             "driver library at run time, so the same NeuralSharp build runs with or without a GPU: without one, "
             "detection reports the reason and everything uses the CPU."),
        h2("24.2 Kernels written as PTX"),
        para("PTX is NVIDIA's portable GPU assembly language (glossary <b>PTX</b>). The class <code>PtxKernels</code> "
             "generates the text for all kernels with C# string templates; most element-wise kernels differ only in "
             "a few instructions, so one template (<code>Elementwise</code>) produces them. The test program can dump "
             "the whole module:"),
        output("""
            dotnet run -c Release --project tests/NeuralSharp.Tests -- --dump-ptx kernels.ptx
            Wrote 104879 characters of PTX to kernels.ptx
            """, caption="Dumping the generated GPU code (57 kernels, 4,087 lines)"),
        code(RELU_PTX, "One generated kernel: ReLU (from kernels.ptx, comments added)", lang="text"),
        reftable(["Kernel family", "Examples"], [
            ["Element-wise (one thread per element, blocks of 256)", "fill, affine, axpy, add/sub/mul, sigmoid, tanh, relu, gelu, exp, log and their backward kernels, dropout"],
            ["Reductions (shared memory and atomic additions)", "sum_f32, sum_rows_f32, sum_axis_f32, norm_stats_f32, group_reduce_f32"],
            ["Matrix product (16×16 shared-memory tiles, batched over grid z)", "matmul_f32"],
            ["Row-wise (one thread per row)", "softmax, log-softmax, argmax, class_match, layernorm_fused, scale_mask_softmax"],
            ["Data movement", "permute, copy2d, gather (embeddings), scatter_add, im2col/col2im, maxpool"],
            ["Optimizers", "sgd_momentum_f32, adam_f32"],
            ["Decoding", "decoder_mask, kv_write, bias_gelu, sample_rows"],
        ], caption="Table 24.2 — The 57 kernels by family"),
        honestbox("Why PTX and not CUDA C",
                  "<p>Compiling CUDA C needs nvcc or NVRTC from the CUDA Toolkit, a multi-gigabyte install the library "
                  "deliberately avoids. PTX is accepted directly by every NVIDIA driver, which compiles it for the card "
                  "in use. The price is that kernels are written at assembly level; the generator keeps that manageable, "
                  "and every kernel is validated with NVIDIA's <code>ptxas</code> for architectures sm_50 to sm_120 "
                  "during development.</p>"),
        h2("24.3 Launching and the stream"),
        para("An operation computes its grid (for element-wise kernels: n / 256 blocks of 256 threads), packs its "
             "arguments (pointers, sizes, scalars as 64-bit slots) and calls <code>cuLaunchKernel</code> on the backend's "
             "stream. The call returns immediately; the GPU runs kernels in stream order. That is why operations look "
             "instantaneous from C#, why a read (<code>Item()</code>, <code>ToArray()</code>) waits, and why telemetry "
             "timings on the GPU measure launch time unless <code>SynchronizeForTiming</code> is set (" + ch("telemetry") + ")."),
        reftable(["Cost", "Typical size", "Consequence"], [
            ["Kernel launch (CPU side)", "a few µs", "Thousands of tiny operations per step are launch-bound; batch them (" + ch("performance") + ") or use a graph"],
            ["Host ↔ device copy", "~10 GB/s + fixed latency", "Upload data once per batch, read results rarely"],
            ["Synchronization", "waits for all queued work", "Avoid in inner loops"],
        ], caption="Table 24.3 — Where GPU time goes besides arithmetic"),
        h2("24.4 Memory"),
        para("<code>cuMemAlloc</code> is slow and synchronizing, so it must not happen in every step. The backend "
             "keeps freed blocks in a dictionary from element count to a stack of device pointers. An allocation of the "
             "same size pops a cached block; only otherwise is the driver asked. Freed blocks are returned under a lock "
             "and never touch the driver, because the finalizer thread may be the one freeing. If the driver reports "
             "out-of-memory, the backend forces a garbage collection, frees the cache and retries once. Limits from "
             "<code>ComputeResources.GpuMemoryLimit</code> are enforced before any allocation (" + ch("devices") + ")."),
        cpugpu("watching the GPU allocator settle",
               """
               // on the CPU the same pattern applies to pooled float[] arrays
               Console.WriteLine(ComputeResources.GetMemoryUsage(Device.Cpu));
               """,
               """
               var gpu = Device.Cuda();
               for (int epoch = 0; epoch < 3; epoch++)
               {
                   trainer.Fit(loader, epochs: 1);
                   Console.WriteLine(ComputeResources.GetMemoryUsage(gpu));   // InUse + Cached stay flat
               }
               ComputeResources.ReleaseCachedMemory(gpu);                    // give cached blocks back
               """),
        h2("24.5 Graphs"),
        para("For a compute graph the backend switches its stream into capture mode (<code>cuStreamBeginCapture</code>, "
             "relaxed mode), runs the step, and ends the capture, which yields a graph of every recorded kernel; "
             "<code>cuGraphInstantiateWithFlags</code> turns it into an executable graph that <code>cuGraphLaunch</code> "
             "replays. Memory freed during recording is kept aside for the graph's exclusive use, since the graph writes to "
             "those addresses on every replay; it returns to the pool only when the graph is disposed. The requirements for "
             "a replayable step are in " + ch("generation") + "."),
        h2("24.6 Numerical differences between CPU and GPU"),
        para("Both devices compute in float32, but not in the same order: GPU reductions sum in parallel trees, some "
             "GPU functions use fast approximate instructions (e.g. <code>ex2.approx</code> for sigmoid and tanh), and "
             "fused multiply-adds round differently from separate operations. Results therefore agree to about 5–6 "
             "significant digits, not bit for bit. The test suite compares the devices with a relative tolerance, and "
             "sampling is designed so that the same seed still picks the same tokens (" + ch("generation") + ")."),
        trap("expecting identical GPU runs",
             "<p>Parallel reductions that use atomic additions finish in a different order from run to run, so two GPU "
             "trainings with the same seeds can differ in the last digits and drift apart over many epochs. For exact "
             "reproducibility, train on the CPU.</p>"),
        practice([
            (1, "Which files must be installed on a machine to use the GPU backend?",
             "Only the NVIDIA display driver (which provides nvcuda.dll or libcuda.so.1). No CUDA Toolkit, cuBLAS or cuDNN."),
            (1, "Why does <code>loss.Item()</code> take longer on the GPU than the whole forward pass seemed to?",
             "The forward pass only queued kernels; <code>Item()</code> waits for all of them to finish and then copies the value."),
            (2, "Dump the PTX and find the kernel used by <code>Tensor.Softmax()</code>. How is one row processed?",
             "<code>softmax_f32</code>, launched with one thread per row: the thread scans its row for the maximum, sums "
             "exp(x − max) and writes each element divided by that sum (subtracting the maximum keeps the exponentials "
             "from overflowing)."),
            (2, "A model uses 2 GB on the GPU after training although only 500 MB is in use. Explain and fix.",
             "The rest is cached blocks kept for reuse (<code>Cached</code> in <code>GetMemoryUsage</code>). Call "
             "<code>ComputeResources.ReleaseCachedMemory(gpu)</code>, or set a <code>GpuMemoryLimit</code>."),
            (3, "Outline the steps to add a new GPU kernel, e.g. a fused bias + ReLU.",
             "Write the PTX with the element-wise template (load x and bias[i % cols], add, max with 0, store), add its name "
             "to the kernel list, get the function handle when the module loads, implement the backend method with "
             "<code>Launch1D</code>, provide the CPU equivalent, and add CPU/GPU comparison and gradient tests (" + ch("testing") + ")."),
        ], PART),
        footer("CUDA", "Driver API", "PTX", "JIT compilation", "Kernel", "Thread block", "Grid", "Stream",
               "Caching allocator", "CUDA graph", "Stream capture"),
    )
