"""Chapter 20 — Performance Tuning."""
from gen import *

PART = "IV"


def build():
    return page(
        chapter_open(
            "performance",
            "NeuralSharp is fast by default, but the same model can train ten times faster or slower depending on "
            "batch size, layer width, device, thread count and a few habits in your own code. This chapter measures "
            "those effects on real runs, gives a checklist, and explains when the GPU pays off.",
            "Always build and run with <code>-c Release</code>.",
            "Batch size is the biggest lever: throughput rose 14× from batch 8 to batch 512 in our measurement.",
            "Cost grows with the square of layer width; measure samples per second when you resize.",
            "Avoid host reads (<code>Item()</code>, <code>ToArray()</code>) inside hot loops, especially on the GPU.",
            "The GPU wins when each step carries large matrix products: wide layers, big batches, convolutions, attention.",
        ),
        h2("20.1 Measuring"),
        para("Measure throughput in samples per second over whole epochs, after a warm-up epoch (the first one pays "
             "for the .NET JIT compiler, GPU code compilation and memory allocation). The Trainer's "
             "<code>EpochCompleted</code> event already reports <code>SamplesPerSecond</code> (" + ch("telemetry") + ")."),
        snippet("""
            double Measure(Trainer trainer, DataLoader loader)
            {
                trainer.Fit(loader, epochs: 1);                   // warm-up
                var clock = Stopwatch.StartNew();
                trainer.Fit(loader, epochs: 2);
                return 2.0 * loader.SampleCount / clock.Elapsed.TotalSeconds;
            }
            """, caption="A throughput helper used for the measurements below"),
        h2("20.2 What the knobs do"),
        para("A 32 → width → width → 1 MLP with Adam, 16,384 rows, on the 4-core cloud CPU used for this book. Change "
             "one setting at a time:"),
        mex("batch size, threads and width", None,
            """
            foreach (int b in new[] { 8, 32, 128, 512, 2048 })   Console.WriteLine($"{b,5}: {Throughput(b, 256, 4),10:N0} samples/s");
            foreach (int th in new[] { 1, 2, 4 })                Console.WriteLine($"{th}: {Throughput(512, 256, th),10:N0} samples/s");
            foreach (int w in new[] { 32, 128, 512 })            Console.WriteLine($"{w,4}: {Throughput(512, w, 4),10:N0} samples/s");
            // Throughput(batch, width, threads) sets ComputeResources.MaxCpuThreads, builds the model,
            // a Trainer and a DataLoader, and returns Measure(trainer, loader)
            """,
            out="""
            batch size (width 256, 4 threads):
                  8:      9,691 samples/s
                 32:     32,788 samples/s
                128:    112,152 samples/s
                512:    137,602 samples/s
               2048:    141,305 samples/s
            threads (batch 512, width 256):
              1:     85,195 samples/s
              2:    134,699 samples/s
              4:    150,105 samples/s
            width (batch 512, 4 threads):
                32:  1,819,181 samples/s
               128:    368,595 samples/s
               512:     44,301 samples/s
            """),
        reftable(["Knob", "Observation", "Why"], [
            ["Batch size", "×14 from 8 to 512, then flat", "Every step has fixed costs (dispatch, allocation, the optimizer update); larger batches spread them over more samples"],
            ["CPU threads", "×1.8 from 1 to 4", "Matrix products and element-wise kernels split across cores; memory bandwidth limits the rest"],
            ["Layer width", "4× wider ≈ 8–16× slower", "A dense layer costs width² per sample"],
        ], caption="Table 20.1 — Reading the measurements"),
        honestbox("Batch size is not free",
                  "<p>Larger batches mean fewer optimizer steps per epoch, so a model may need more epochs (or a higher "
                  "learning rate) to reach the same loss. For most problems batches of 32–256 balance speed and "
                  "learning; measure time to reach a target validation loss, not only samples per second.</p>"),
        h2("20.3 CPU or GPU, quantitatively"),
        reftable(["Workload", "Typical winner", "Reason"], [
            ["MLP under ~100k parameters, batch ≤ 256", "CPU", "Each GPU kernel launch costs a few µs; the work per launch is tiny"],
            ["MLP with 512+ units, batch ≥ 256", "GPU", "Large matrix products"],
            ["CNN on images", "GPU", "im2col + GEMM is ideal GPU work"],
            ["LSTM/GRU, hidden ≥ 128, batch ≥ 64", "GPU", "Per-step products are large enough"],
            ["LSTM/GRU, small hidden or batch", "CPU", "Many small sequential steps"],
            ["Transformer training", "GPU", "Attention and feed-forward products; the " + ch("gpt") + " GPT trains about 8× faster on a laptop RTX 5050"],
            ["Single-sample inference, latency-critical", "CPU", "No transfers, no launches"],
            ["Batched inference (many requests)", "GPU", "Throughput"],
        ], caption="Table 20.2 — Where each device shines"),
        cpugpu("making the GPU busy",
               """
               ComputeResources.MaxCpuThreads = Environment.ProcessorCount;
               var loader = new DataLoader(train, batchSize: 64, shuffle: true);
               """,
               """
               ComputeResources.MaxCpuThreads = 4;           // CPU only gathers batches now
               var loader = new DataLoader(train, batchSize: 512, shuffle: true, device: Device.Cuda());
               // larger batches -> fewer, larger kernels; consider a higher learning rate
               """),
        h2("20.4 Habits that cost speed"),
        reftable(["Habit", "Cost", "Instead"], [
            ["<code>loss.Item()</code> every step", "One GPU synchronization per step", "Read every N steps, or let the Trainer accumulate on the device"],
            ["Creating tensors from host arrays every step", "An upload per step", "Build the dataset once; use the DataLoader"],
            ["No <code>TensorScope</code> in the loop", "Memory churn, finalizer pauses (" + ch("memory") + ")", "Scope every loop body"],
            ["Debug build", "Several times slower", "<code>dotnet run -c Release</code>"],
            ["Batch- or operation-level telemetry left on", "Synchronizations and per-operation events", "<code>Training</code> level in production"],
            ["Host-side metrics (" + ch("metrics") + ")", "A synchronization per batch", "Tensor-only metrics"],
            ["Recomputing the whole text for every generated token", "O(n²) work", "KV cache (" + ch("generation") + ")"],
        ], caption="Table 20.3 — Common slow-downs"),
        h2("20.5 Number types"),
        para("All tensors store and compute float32. <code>Tensor.From&lt;T&gt;</code> converts other types on the way "
             "in and <code>ToArray&lt;T&gt;</code> on the way out (" + ch("tensors") + "), so <code>double</code>, "
             "<code>int</code>, <code>byte</code>, <code>Half</code> or <code>decimal</code> data can be used directly. "
             "Float32 has about 7 significant digits, which is ample for learning; keep large-valued targets (prices in "
             "the millions) scaled, both for learning and for precision."),
        honestbox("No half-precision training",
                  "<p>The kernels compute in float32 only; there is no float16/bfloat16 mixed-precision mode, which on "
                  "recent GPUs could roughly double throughput for large models. For the model sizes in this book "
                  "float32 is fast enough.</p>"),
        h2("20.6 A checklist"),
        deriv("Before blaming the library", [
            "Release build? Warm-up excluded from the timing?",
            "Batch size at least 32 (CPU) or 128–512 (GPU)?",
            "Loop bodies inside a <code>TensorScope</code>; no <code>Item()</code> per step?",
            "Telemetry at <code>Training</code> level only?",
            "On the GPU: data loaded by a <code>DataLoader</code> on the GPU device, not uploaded by hand each step?",
            "For generation: KV cache on, and on the GPU CUDA graphs on (" + ch("generation") + ")?",
            "For latency: model in evaluation mode once, <code>Predict</code> on batches when possible?",
        ]),
        practice([
            (1, "Your GPU training runs at the same speed as on the CPU with batch 16. What do you try first?",
             "Increase the batch size (e.g. 256) and check that nothing reads values back every step."),
            (1, "Why does doubling a hidden layer's width roughly quadruple its cost?",
             "A Linear(w, w) layer does w·w multiply-adds per sample; doubling w multiplies that by 4."),
            (2, "Measure your own machine: run the Section 20.2 experiment and find the batch size beyond which throughput stops improving.",
             "Throughput rises steeply at first and flattens once fixed per-step costs are amortized; on the book's 4-core "
             "machine that was around 512. More cores or a GPU push the knee to larger batches."),
            (2, "A training loop prints the loss every step on the GPU. How do you keep the output but restore speed?",
             "Print every 50 steps, or accumulate the losses into a device tensor (like the Trainer) and read it once "
             "per epoch; or use the Trainer with a <code>ConsoleLogger</code> at <code>Training</code> level."),
            (3, "Find the smallest MLP width at which your GPU beats your CPU for batch 256.",
             "Run the width experiment on both devices (create the model and loader with <code>device:</code>) for widths "
             "64, 128, 256, 512, 1024 and compare samples per second; the crossover is typically a few hundred units."),
        ], PART),
        footer("Throughput", "Latency", "Batch size", "Warm-up", "Kernel launch", "Synchronization",
               "Float32", "Mixed precision", "Release build"),
    )
