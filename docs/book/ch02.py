"""Chapter 2 — Devices, Backends and Compute Resources."""
from gen import *

PART = "I"


def backends_svg():
    w, h = 480, 190
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    def box(x, y, bw, bh, label, sub, fill="#ffffff", stroke="#0f6b5c", color="#0f6b5c"):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        p.append(svg_text(x + bw / 2, y + 17, label, 9.5, color))
        if sub:
            p.append(svg_text(x + bw / 2, y + 31, sub, 7.6, "#56606a"))

    def arrow(x1, y1, x2, y2):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#56606a" stroke-width="1.1"/>')
        p.append(f'<polygon points="{x2},{y2} {x2 - 4},{y2 - 7} {x2 + 4},{y2 - 7}" fill="#56606a"/>')

    box(150, 6, 180, 40, "your code", "Tensor · Module · Optimizer", "#e6f2ef")
    arrow(240, 46, 240, 64)
    box(150, 66, 180, 40, "Device", "Device.Cpu · Device.Cuda(n)")
    arrow(200, 106, 110, 128)
    arrow(280, 106, 370, 128)
    box(20, 130, 180, 52, "CPU backend", "SIMD Vector&lt;float&gt; · all cores")
    box(280, 130, 180, 52, "CUDA backend", "driver API · PTX → GPU")
    p.append(svg_text(110, 176, "pooled float[] memory", 7.4, "#56606a"))
    p.append(svg_text(370, 176, "caching GPU allocator · stream", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "devices",
            "Every tensor lives on a device, and every operation runs on the device of its inputs. This chapter "
            "explains the two devices NeuralSharp supports, how to find and choose them, how GPU work is queued, "
            "and how to control how much of the machine the library may use: CPU threads and memory on each device.",
            "<code>Device.Cpu</code> is always there; <code>Device.Cuda(n)</code> needs only the NVIDIA display driver.",
            "<code>Device.Default</code> is read when a tensor or layer is created. Explicit <code>device:</code> arguments win over it.",
            "GPU work is queued: timing needs <code>device.Synchronize()</code>; reading a value (<code>Item()</code>, <code>ToArray()</code>) waits automatically.",
            "<code>ComputeResources</code> caps CPU threads and per-device memory, and reports memory use.",
            "Pick the CPU for small models and low latency, the GPU for wide layers, big batches, convolutions and transformers.",
        ),
        h2("2.1 One library, two backends"),
        para("Your code only ever talks to tensors, modules and optimizers. Each tensor carries a <code>Device</code>, "
             "and the device owns a <b>backend</b>: the object that actually allocates memory and runs the arithmetic. "
             "The backends are internal, so the same program runs on either without changes."),
        diagram("Figure 2.1 — How a call reaches the hardware", backends_svg(),
                "The device of the input tensors decides which backend runs an operation."),
        reftable(["", "CPU backend", "CUDA backend"], [
            ["Hardware", "Any x64 or Arm64 CPU", "NVIDIA GPU, compute capability 5.0 (Maxwell) or newer"],
            ["Needs", "Nothing", "The NVIDIA display driver (nvcuda.dll / libcuda.so.1); no CUDA Toolkit"],
            ["Speed comes from", "SIMD (<code>Vector&lt;float&gt;</code>), cache-blocked matrix products, all cores", "Thousands of GPU threads; kernels written in PTX by the library and compiled by the driver"],
            ["Memory", "Pooled .NET arrays", "GPU memory from a caching allocator"],
            ["Work is", "Done when the call returns", "Queued on a stream; runs asynchronously"],
            ["Best for", "Small models, small batches, low-latency single predictions", "Wide layers, large batches, CNNs, LSTMs, transformers"],
        ], caption="Table 2.1 — The two backends"),
        defbox("Backend", "<p>The internal engine behind a device. It implements a fixed set of primitives (fill, copy, "
               "element-wise functions, matrix products, reductions, softmax, convolution helpers, optimizer updates) "
               "on contiguous row-major float32 buffers. Everything else in the library is built from these primitives, "
               "which is why every feature works on both devices. " + ch("cpu") + " and " + ch("cuda") + " open them up.</p>"),
        h2("2.2 Finding the GPUs"),
        para("Detection is lazy and safe: the first time you touch <code>Device.IsCudaAvailable</code>, "
             "<code>Device.CudaDeviceCount</code> or <code>Device.Default</code>, the library tries to load the driver. "
             "If that fails, CUDA is simply reported as unavailable. Only <code>Device.Cuda(n)</code> throws, and its "
             "message says why."),
        mex("why is there no GPU?", None,
            """
            using NeuralSharp;

            try
            {
                var gpu = Device.Cuda(0);
                Console.WriteLine($"using {gpu.Name}");
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine(ex.Message);                 // the reason, in plain words
            }
            """,
            out="""
            CUDA is not available: the NVIDIA driver library (libcuda.so.1 / libcuda.so) was not found
            """,
            after="On a machine with a GPU the same program prints, for example, <code>using NVIDIA GeForce RTX 5050 "
                  "Laptop GPU (8150 MiB, 20 SMs)</code>: the model, its memory and its number of streaming multiprocessors."),
        reftable(["Message ends with", "Meaning", "Fix"], [
            ["<i>driver library … was not found</i>", "No NVIDIA driver installed (or a non-NVIDIA GPU)", "Install the NVIDIA display driver, or use the CPU"],
            ["<i>cuInit failed with error N</i>", "Driver present but no usable GPU (e.g. a container without GPU access)", "Pass the GPU through (<code>docker run --gpus all</code>) or use the CPU"],
            ["<i>reports no CUDA devices</i>", "Driver loaded, zero GPUs visible", "Check <code>CUDA_VISIBLE_DEVICES</code> and <code>nvidia-smi</code>"],
            ["<i>disabled by the NEURALSHARP_DISABLE_CUDA …</i>", "You switched the GPU off on purpose", "Unset the variable"],
            ["<i>device N does not exist</i>", "The ordinal is larger than the number of GPUs", "Use <code>Device.CudaDeviceCount - 1</code> or less"],
        ], caption="Table 2.2 — Reading the CUDA error message"),
        h2("2.3 Choosing the device"),
        para("There are exactly two rules. First, an explicit <code>device:</code> argument (on a tensor, a layer, a "
             "loader) always wins. Second, when none is given, <code>Device.Default</code> is used <i>at the moment the "
             "object is created</i>. Changing <code>Device.Default</code> later does not move anything that already exists: "
             "use <code>.To(device)</code> for that."),
        cpugpu("the one line that selects the device",
               """
               Device.Default = Device.Cpu;
               """,
               """
               Device.Default = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
               """,
               "Put this line first in <code>Main</code> (or in your host's startup code). Everything created afterwards follows it."),
        para("Real applications usually let the user decide. The helper below understands the same values as all the "
             "sample projects (<code>auto</code>, <code>cpu</code>, <code>cuda</code>, <code>cuda:1</code>) and can "
             "be copied into any project, reading from the command line, an environment variable or a configuration file."),
        snippet("""
            static Device ParseDevice(string? text) => text?.Trim().ToLowerInvariant() switch
            {
                null or "" or "auto" => Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu,
                "cpu" => Device.Cpu,
                "cuda" or "gpu" => Device.Cuda(0),
                var s when s.StartsWith("cuda:") => Device.Cuda(int.Parse(s[5..])),
                var s => throw new ArgumentException($"unknown device '{s}' (use auto, cpu, cuda or cuda:N)"),
            };

            // e.g.  dotnet run -- --device cuda:1     or     MYAPP_DEVICE=cpu dotnet run
            string? choice = args.SkipWhile(a => a != "--device").Skip(1).FirstOrDefault()
                             ?? Environment.GetEnvironmentVariable("MYAPP_DEVICE");
            Device.Default = ParseDevice(choice);
            Console.WriteLine($"running on {Device.Default} ({Device.Default.Name})");
            """, caption="A reusable device switch"),
        reftable(["Member", "What it gives you"], [
            ["<code>Device.Cpu</code>", "The CPU device (always available)."],
            ["<code>Device.Cuda(int ordinal = 0)</code>", "GPU number <i>ordinal</i>; throws <code>InvalidOperationException</code> with the reason when it does not exist."],
            ["<code>Device.Default</code>", "Get/set the device used when none is given. Starts as <code>cuda:0</code> if a GPU exists, else <code>cpu</code>."],
            ["<code>Device.IsCudaAvailable</code>", "True when the driver loaded and at least one GPU was found."],
            ["<code>Device.CudaDeviceCount</code>", "Number of GPUs (0 without a driver)."],
            ["<code>device.Type</code>, <code>device.Ordinal</code>", "<code>DeviceType.Cpu</code> or <code>DeviceType.Cuda</code>, and the GPU index."],
            ["<code>device.Name</code>", "Human-readable description (CPU threads and SIMD width, or GPU model, memory and SMs)."],
            ["<code>device.Synchronize()</code>", "Blocks until all queued work on the device has finished."],
            ["<code>device.ToString()</code>", "<code>cpu</code> or <code>cuda:N</code>, the same spelling the samples accept."],
        ], caption="Table 2.3 — The Device API"),
        h2("2.4 Several GPUs, or CPU and GPU together"),
        para("Each GPU is a separate device with its own memory. Tensors on different devices cannot be combined in one "
             "operation; move data with <code>.To(device)</code>, which copies it. Two independent models can run on two "
             "GPUs (or one on the GPU and one on the CPU) at the same time, each from its own thread."),
        snippet("""
            var gpu0 = Device.Cuda(0);
            var gpu1 = Device.Cuda(1);

            using var modelA = BuildModel(); modelA.To(gpu0);      // BuildModel(): your own factory
            using var modelB = BuildModel(); modelB.To(gpu1);

            var work = new[]
            {
                Task.Run(() => Train(modelA, gpu0)),                 // each thread uses one device
                Task.Run(() => Train(modelB, gpu1)),
            };
            await Task.WhenAll(work);
            """, caption="Two models trained side by side on two GPUs"),
        honestbox("What the library does not do for you",
                  "<p>There is no automatic splitting of one model or one batch across several GPUs (data or model "
                  "parallelism). If you need it, train copies of the model on different devices and average their "
                  "parameters yourself, or split the model's layers across devices and move the activations between "
                  "them with <code>.To(device)</code>. For almost all projects in this book a single GPU is ample.</p>"),
        h2("2.5 GPU work is asynchronous"),
        para("A GPU operation returns as soon as its kernel is <i>queued</i>, not when it has finished. This lets the CPU "
             "prepare the next operation while the GPU computes. Anything that reads data back to .NET (<code>Item()</code>, "
             "<code>ToArray()</code>, <code>ToString()</code>, <code>Predict(float[,])</code>) waits for the result "
             "automatically, so ordinary code never notices. Only timing needs care: call <code>Synchronize()</code> "
             "before starting and before stopping the clock."),
        mex("measuring throughput correctly", None,
            """
            using System.Diagnostics;
            using NeuralSharp;

            static double MatMulGflops(Device device, int n, int repeats)
            {
                using var a = Tensor.Uniform([n, n], -1, 1, new Random(1), device);
                using var b = Tensor.Uniform([n, n], -1, 1, new Random(2), device);
                using (a.MatMul(b)) { }                 // warm-up: first call compiles/allocates
                device.Synchronize();
                var clock = Stopwatch.StartNew();
                for (int i = 0; i < repeats; i++)
                {
                    using var c = a.MatMul(b);
                }
                device.Synchronize();                   // wait for the queue to drain
                return 2.0 * n * n * n * repeats / clock.Elapsed.TotalSeconds / 1e9;
            }

            Console.WriteLine($"cpu : {MatMulGflops(Device.Cpu, 1024, 10):F1} GFLOP/s");
            if (Device.IsCudaAvailable)
                Console.WriteLine($"cuda: {MatMulGflops(Device.Cuda(), 1024, 10):F1} GFLOP/s");
            """,
            out="""
            cpu : 65.5 GFLOP/s
            """,
            after="The CPU figure is from a 4-core cloud machine; a desktop CPU with more cores reaches several times "
                  "more, and a GPU several times more again. The formula counts one multiply and one add for each of "
                  "the n³ inner-product terms."),
        trap("timing without Synchronize",
             "<p>Without the second <code>Synchronize()</code> the stopwatch measures only how fast kernels are queued, "
             "which makes the GPU look impossibly fast. Without the warm-up, the first call's one-off costs (compiling "
             "the GPU code, allocating memory) are included.</p>"),
        h2("2.6 Compute resources: threads and memory"),
        para("By default NeuralSharp uses every CPU core and allocates memory as the network needs it. When the library "
             "shares a machine with other work (a web server, a desktop UI, other models), cap it with the static "
             "<code>ComputeResources</code> class. Set the limits once at startup, before heavy work begins."),
        reftable(["Member", "Default", "Effect"], [
            ["<code>MaxCpuThreads</code>", "all cores", "Threads used by CPU kernels, data loading and CSV parsing; 1 makes all CPU work single-threaded."],
            ["<code>CpuMemoryLimit</code>", "<code>null</code> (unlimited)", "Maximum bytes of tensor memory on the CPU."],
            ["<code>GpuMemoryLimit</code>", "<code>null</code> (whole card)", "Maximum bytes of tensor memory on each GPU."],
            ["<code>GetMemoryUsage(device)</code>", "—", "A <code>MemoryUsage</code>: <code>InUse</code>, <code>Cached</code>, <code>Limit</code>, <code>Reserved</code>."],
            ["<code>ReleaseCachedMemory(device?)</code>", "—", "Returns cached blocks to the system (one device, or all when null)."],
        ], caption="Table 2.4 — ComputeResources"),
        mex("limits and usage",
            "Memory freed by a tensor is kept in a cache for reuse (" + ch("memory") + "). The limit counts memory "
            "<i>in use</i>; the cache is released automatically when a new allocation would otherwise exceed the limit.",
            """
            ComputeResources.MaxCpuThreads = 4;                        // leave the rest for the app
            ComputeResources.CpuMemoryLimit = 64L * 1024 * 1024;       // 64 MiB

            using (var scope = new TensorScope())
            {
                for (int i = 0; i < 3; i++) Tensor.Zeros([1024, 1024], Device.Cpu);   // 4 MiB each
                Console.WriteLine(ComputeResources.GetMemoryUsage(Device.Cpu));
            }
            Console.WriteLine(ComputeResources.GetMemoryUsage(Device.Cpu));
            ComputeResources.ReleaseCachedMemory();
            Console.WriteLine(ComputeResources.GetMemoryUsage(Device.Cpu));

            try
            {
                using var huge = Tensor.Zeros([8192, 4096], Device.Cpu);   // 128 MiB
            }
            catch (ResourceLimitExceededException ex)
            {
                Console.WriteLine(ex.Message);
            }
            """,
            out="""
            in use 12.0 MiB, cached 0 B, limit 64.0 MiB
            in use 0 B, cached 12.0 MiB, limit 64.0 MiB
            in use 0 B, cached 0 B, limit 64.0 MiB
            Allocating 134,217,728 bytes on the CPU would exceed its memory limit of 67,108,864 bytes (0 in use). Raise the limit in ComputeResources, use smaller batches, or dispose tensors you no longer need.
            """),
        cpugpu("capping resources in a shared application",
               """
               ComputeResources.MaxCpuThreads = Math.Max(1, Environment.ProcessorCount / 2);
               ComputeResources.CpuMemoryLimit = 2L << 30;               // 2 GiB
               Device.Default = Device.Cpu;
               """,
               """
               ComputeResources.MaxCpuThreads = 2;                       // the CPU only feeds the GPU
               ComputeResources.GpuMemoryLimit = 3L << 30;               // 3 GiB of the card
               Device.Default = Device.Cuda();
               Console.WriteLine(ComputeResources.GetMemoryUsage(Device.Default));
               """,
               "On the GPU the CPU threads matter much less (they mostly load data), so a small number is fine."),
        trap("setting a limit below what one step needs",
             "<p>A limit is a hard wall: the allocation that would cross it throws "
             "<code>ResourceLimitExceededException</code>. Catch it where you choose the batch size and retry with a "
             "smaller batch, or raise the limit. It does not slow the program down or fall back to the CPU.</p>"),
        h2("2.7 CPU or GPU? A decision table"),
        reftable(["Your situation", "Choose", "Why"], [
            ["Fewer than ~10,000 parameters, or batches of a few rows", "CPU", "GPU launch cost (a few µs per operation) dominates"],
            ["Single predictions with strict latency (&lt; 1 ms)", "CPU", "No transfer to/from the GPU"],
            ["Dense layers of 512+ units with batches of 64+", "GPU", "Matrix products scale with the GPU's thousands of cores"],
            ["Convolutions, LSTMs over long sequences, transformers", "GPU", "Large, regular arithmetic per step"],
            ["Text generation with a KV cache and CUDA graphs", "GPU (large models) / either (tiny ones)", "See " + ch("generation")],
            ["No NVIDIA GPU, or a server without one", "CPU", "Everything works; only speed differs"],
            ["Train big, serve small", "GPU to train, CPU to serve", "Model files are device-independent"],
        ], caption="Table 2.5 — Picking a device"),
        trap("creating the data on a different device from the model",
             "<p><i>Tensors are on different devices (cpu and cuda:0)</i> means an input was created before "
             "<code>Device.Default</code> was set, or with an explicit different device. Create inputs with "
             "<code>device: model's device</code> or call <code>input.To(device)</code>.</p>"),
        practice([
            (1, "Write a program that prints every device on the machine with its name, one per line.",
             "Print <code>Device.Cpu.Name</code>, then loop <code>for (int i = 0; i &lt; Device.CudaDeviceCount; i++)</code> "
             "and print <code>Device.Cuda(i)</code> and <code>Device.Cuda(i).Name</code>."),
            (1, "Make your XOR program from " + ch("start") + " accept <code>--device</code> using <code>ParseDevice</code>.",
             "Copy <code>ParseDevice</code>, read the value after <code>--device</code> from <code>args</code>, and assign "
             "<code>Device.Default</code> before any tensor or layer is created."),
            (2, "Measure MatMul throughput for n = 128, 512 and 2048 on each device you have. Where does the GPU overtake the CPU?",
             "Call <code>MatMulGflops</code> for each size. Small n is launch-bound on the GPU; typically the GPU is "
             "ahead from a few hundred rows and far ahead at 2048."),
            (2, "Set <code>CpuMemoryLimit</code> to 32 MiB and allocate 4 MiB tensors in a loop without disposing them. After how many does it throw, and why exactly then?",
             "After 8 allocations (8 × 4 MiB = 32 MiB in use) the ninth would exceed the limit. The cache does not count "
             "because nothing was freed."),
            (3, "Train two copies of the XOR network at the same time, one on the CPU and one on the GPU, each on its own thread, and print both losses.",
             "Build each model and its tensors with an explicit <code>device:</code> (not <code>Device.Default</code>, "
             "which is shared), run each training loop in <code>Task.Run</code>, and <code>await Task.WhenAll</code>. "
             "TensorScope and NoGrad are per thread, so the loops do not interfere."),
        ], PART),
        footer("Device", "Backend", "CUDA", "PTX", "Streaming multiprocessor", "Kernel", "Synchronize",
               "SIMD", "ComputeResources", "Memory limit", "Asynchronous execution", "GFLOP/s"),
    )
