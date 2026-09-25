"""Chapter 4 — Memory and Tensor Lifetime."""
from gen import *

PART = "I"


def lifetime_svg():
    w, h = 480, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    def box(x, y, bw, bh, label, sub, fill):
        p.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="5" fill="{fill}" stroke="#0f6b5c" stroke-width="1.2"/>')
        p.append(svg_text(x + bw / 2, y + 18, label, 9.2, "#0f6b5c"))
        p.append(svg_text(x + bw / 2, y + 32, sub, 7.4, "#56606a"))

    def arrow(x1, y1, x2, y2, label=None, ly=0):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#56606a" stroke-width="1.1"/>')
        p.append(f'<polygon points="{x2},{y2} {x2 - 7},{y2 - 4} {x2 - 7},{y2 + 4}" fill="#56606a"/>')
        if label:
            p.append(svg_text((x1 + x2) / 2, y1 - 6 + ly, label, 7.4, "#56606a"))

    box(10, 50, 110, 44, "system memory", "cuMemAlloc / new float[]", "#ffffff")
    box(185, 50, 110, 44, "tensor in use", "counted in InUse", "#e6f2ef")
    box(360, 50, 110, 44, "cache (pool)", "counted in Cached", "#fff4e2")
    arrow(120, 64, 185, 64, "first allocation")
    arrow(295, 64, 360, 64, "Dispose / scope end")
    p.append('<path d="M 415 50 C 415 12, 240 12, 240 48" fill="none" stroke="#0f6b5c" stroke-width="1.1"/>')
    p.append('<polygon points="240,50 236,42 244,42" fill="#0f6b5c"/>')
    p.append(svg_text(328, 20, "same-size allocation reuses a cached block", 7.6, "#0f6b5c"))
    p.append('<path d="M 415 94 C 415 138, 65 138, 65 96" fill="none" stroke="#56606a" stroke-width="1.1"/>')
    p.append('<polygon points="65,94 61,102 69,102" fill="#56606a"/>')
    p.append(svg_text(240, 132, "ReleaseCachedMemory() or a memory limit gives blocks back", 7.6, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "memory",
            "A tensor owns a block of device memory. On the GPU that memory is scarce and invisible to the .NET "
            "garbage collector, so NeuralSharp gives you explicit, cheap control over when it is returned. This chapter "
            "explains the life of a block, the three ways a tensor is freed, TensorScope in depth, and the patterns for "
            "training loops, evaluation, helper methods and long-running services.",
            "Freed blocks go to a per-device <b>cache</b> and are reused by the next allocation of the same size: after the first step, a training loop allocates nothing new.",
            "A tensor is freed by <code>Dispose()</code> (or <code>using</code>), by the end of the <code>TensorScope</code> that created it, or, late, by the finalizer.",
            "Parameters, gradients and optimizer state are never captured by a scope; <code>Predict</code> cleans up after itself.",
            "Watch <code>ComputeResources.GetMemoryUsage(device).InUse</code>: if it grows every step, something is not being freed.",
            "Scopes belong to one thread; do not keep one open across an <code>await</code>.",
        ),
        h2("4.1 The life of a memory block"),
        para("When a tensor is created, its device's backend looks in the cache for a free block of exactly the right "
             "size. Only if there is none does it ask the system (<code>cuMemAlloc</code> on the GPU, a new "
             "<code>float[]</code> on the CPU). When the tensor is freed, the block goes back to the cache, not to the "
             "system. Because a training loop creates the same shapes on every step, it stops allocating after the first one."),
        diagram("Figure 4.1 — Allocation, use, caching and release", lifetime_svg(),
                "InUse + Cached = Reserved: what the library holds from the system on that device."),
        defbox("Reference-counted storage",
               "<p>A block may be shared: <code>Reshape</code>, <code>Flatten</code> and <code>Detach</code> return new "
               "tensors that point at the same data. The block returns to the cache only when the last tensor using it "
               "is freed, so disposing the original while a view is alive is safe.</p>"),
        mex("views keep the data alive", None,
            """
            var a = Tensor.From([1f, 2, 3, 4, 5, 6]);
            var view = a.Reshape(2, 3);
            a.Dispose();
            Console.WriteLine(view.Sum().Item());     // still valid
            view.Dispose();                           // now the block returns to the cache

            try { a.Sum(); }
            catch (ObjectDisposedException ex) { Console.WriteLine(ex.GetType().Name); }
            """,
            out="""
            21
            ObjectDisposedException
            """),
        h2("4.2 Three ways a tensor is freed"),
        reftable(["Way", "When", "Use it for"], [
            ["<code>Dispose()</code> / <code>using var</code>", "Immediately, at a point you choose", "Tensors you create once and keep: datasets, fixed inputs, results you return"],
            ["End of the <code>TensorScope</code> that was active when the tensor was created", "When the scope is disposed", "Everything created inside one step of a loop"],
            ["Finalizer", "Whenever the garbage collector gets to it: seconds later, or never under low CPU-memory pressure", "Nothing: it is a safety net, not a strategy"],
        ], caption="Table 4.1 — How memory comes back"),
        honestbox("Why the finalizer is not enough",
                  "<p>The garbage collector reacts to pressure on <i>.NET</i> memory. A tensor object is tiny on the .NET "
                  "heap even when it owns 100 MB of GPU memory, so the collector sees no reason to run and the GPU fills "
                  "up. When a GPU allocation does fail, the CUDA backend forces a collection, empties its cache and "
                  "retries once, which rescues many programs, but only after a long pause. Dispose or scope everything "
                  "that is created repeatedly.</p>"),
        h2("4.3 TensorScope in depth"),
        para("A <code>TensorScope</code> registers itself as the current scope of the thread that creates it. Every "
             "tensor created on that thread while it is current is added to its list; when the scope is disposed it "
             "disposes them all and the previous scope becomes current again. Scopes nest."),
        reftable(["Captured by a scope", "Never captured"], [
            ["Results of operations (<code>a + b</code>, <code>MatMul</code>, <code>Forward</code>, losses)", "Layer parameters and buffers (created by the layer)"],
            ["Tensors from <code>Tensor.From</code>, <code>Zeros</code>, <code>Uniform</code>, … created inside it", "Gradients (<code>.Grad</code>) of any tensor"],
            ["Views from <code>Reshape</code>, <code>Detach</code>, <code>To</code> (when it copies)", "Optimizer state (moments, velocities)"],
            ["", "Tensors created before the scope, or on another thread"],
        ], caption="Table 4.2 — What a scope owns"),
        mex("a scoped step against a leaky one",
            "The same computation, first inside a scope and then without one. The usage printed is the CPU "
            "device's; on the GPU the numbers are the same.",
            """
            static string Usage() => ComputeResources.GetMemoryUsage(Device.Cpu).ToString();
            using var data = Tensor.Uniform([256, 256], -1, 1, new Random(1));

            for (int step = 1; step <= 3; step++)
            {
                using var scope = new TensorScope();
                var y = data.MatMul(data).Tanh().Mean();
                Console.WriteLine($"scoped step {step}: {Usage()}");
            }
            Console.WriteLine($"after the loop  : {Usage()}");

            for (int step = 1; step <= 3; step++)
            {
                var y = data.MatMul(data).Tanh().Mean();          // nobody frees these
                Console.WriteLine($"leaky  step {step}: {Usage()}");
            }
            """,
            out="""
            scoped step 1: in use 768.0 KiB, cached 0 B
            scoped step 2: in use 768.0 KiB, cached 0 B
            scoped step 3: in use 768.0 KiB, cached 0 B
            after the loop  : in use 256.0 KiB, cached 512.0 KiB
            leaky  step 1: in use 768.0 KiB, cached 0 B
            leaky  step 2: in use 1.3 MiB, cached 0 B
            leaky  step 3: in use 1.8 MiB, cached 0 B
            """,
            after="In the scoped loop, usage is flat: each step reuses the previous step's blocks. After the loop, "
                  "only <code>data</code> (256 KiB) is in use and the step's two 256 KiB blocks sit in the cache. "
                  "The leaky loop grows by two blocks per step."),
        h3("Keeping one result: scope.Keep"),
        para("A helper that builds a result from temporaries can open its own scope and hand the result out with "
             "<code>scope.Keep(result)</code>. The kept tensor moves to the enclosing scope (if any), so the caller "
             "treats it like any tensor it created itself."),
        mex("a helper that cleans up after itself", None,
            """
            static Tensor Standardize(Tensor x)
            {
                using var scope = new TensorScope();
                var centered = x - x.Mean().Item();
                var result = centered * (1f / MathF.Sqrt((centered * centered).Mean().Item()));
                return scope.Keep(result);                 // everything else is freed here
            }

            using var z = Standardize(Tensor.From([1f, 2, 3, 4]));
            Console.WriteLine(z);
            """,
            out="""
            Tensor(shape=[4], device=cpu)
            [-1.3416, -0.4472, 0.4472, 1.3416]
            """),
        h2("4.4 Patterns"),
        cpugpu("the training step",
               """
               Device.Default = Device.Cpu;
               // model, optimizer, x, y created once, with using
               for (int step = 0; step < steps; step++)
               {
                   using var scope = new TensorScope();
                   var loss = Losses.MeanSquaredError(model.Forward(x), y);
                   optimizer.ZeroGrad();
                   loss.Backward();
                   optimizer.Step();
               }
               """,
               """
               Device.Default = Device.Cuda();
               // identical loop: the scope returns GPU blocks to the GPU cache,
               // so after step 1 no cuMemAlloc is called at all
               """,
               "The loss can be read inside the scope (<code>loss.Item()</code>); after the scope it is disposed."),
        snippet("""
            float EvaluateLoss(Module model, Tensor x, Tensor y)
            {
                using var scope = new TensorScope();
                using var noGrad = Autograd.NoGrad();          // no graph: less memory, faster
                model.Eval();
                var loss = Losses.MeanSquaredError(model.Forward(x), y).Item();
                model.Train();
                return loss;                                   // a float leaves the scope, not a tensor
            }
            """, caption="Evaluation: scope + NoGrad, return plain numbers"),
        snippet("""
            // Load once at startup and switch to evaluation mode once.
            var model = BuildModel();
            model.Load("model.weights");
            model.Eval();

            float[,] Score(float[,] rows) => model.Predict(rows);   // allocates and frees everything itself
            """, caption="A long-running service (web API, worker, desktop app)"),
        reftable(["Situation", "Pattern"], [
            ["Loop body (training, evaluation, generation step)", "<code>using var scope = new TensorScope();</code> first line"],
            ["Helper that returns a tensor", "Own scope + <code>return scope.Keep(result);</code>"],
            ["Helper that returns numbers", "Own scope; return <code>Item()</code> / <code>ToArray()</code> results"],
            ["Data loaded once", "<code>using var</code> at the top of the program, or a field disposed with its owner"],
            ["Inference in a service", "<code>model.Predict(float[,])</code> or <code>using var y = model.Predict(x);</code>"],
            ["Class that owns tensors", "Implement <code>IDisposable</code> and dispose the tensors, model and optimizer"],
        ], caption="Table 4.3 — Which pattern where"),
        h2("4.5 Diagnosing memory problems"),
        reftable(["Symptom", "Likely cause", "Fix"], [
            ["<code>InUse</code> grows every step", "A loop body without a scope, or tensors stored in a list across steps", "Add a scope; store numbers, not tensors"],
            ["<code>ObjectDisposedException</code>", "A tensor used after its scope ended", "Create it outside the loop, or <code>scope.Keep</code> it"],
            ["<i>must be disposed in the reverse order … on the thread that created them</i>", "A scope disposed on another thread (after an <code>await</code>) or out of order", "Keep scopes inside synchronous code"],
            ["<code>ResourceLimitExceededException</code>", "A configured limit (" + ch("devices") + ") is too low for one step", "Smaller batch, or a higher limit"],
            ["CUDA error 2 (out of memory)", "The card is full: batch or model too large, or leaked tensors", "Check <code>InUse</code>; reduce the batch; free leaks"],
            ["Long pauses on the GPU", "Memory reclaimed only through the finalizer", "Dispose or scope repeatedly created tensors"],
        ], caption="Table 4.4 — Memory troubleshooting"),
        trap("keeping a scope open across await",
             "<p>A scope belongs to the thread that created it. After an <code>await</code>, the method may continue on a "
             "different thread: tensors created there are not captured, and disposing the scope there throws. Do the "
             "tensor work in a synchronous method (wrapped in <code>Task.Run</code> if needed) and await outside it.</p>"),
        trap("disposing what the graph still needs",
             "<p>Never dispose a forward result or a loss before <code>Backward()</code> has run: the backward pass reads "
             "those values. Let the step's scope free them after <code>optimizer.Step()</code>.</p>"),
        trap("storing losses as tensors",
             "<p><code>history.Add(loss)</code> keeps every step's loss tensor (and, before Backward, its whole graph). "
             "Store <code>loss.Item()</code> instead, a float.</p>"),
        practice([
            (1, "Print <code>ComputeResources.GetMemoryUsage(device)</code> every 100 steps of your XOR training loop. Is it flat?",
             "Yes: with the <code>TensorScope</code> in the loop body, <code>InUse</code> stays constant after the first step "
             "(parameters, gradients, optimizer state and the four input rows)."),
            (1, "Remove the scope from the XOR loop and run 2,000 epochs. What happens to <code>InUse</code>?",
             "It grows linearly with the number of steps: each step's forward, loss and intermediate tensors are never "
             "freed until the finalizer eventually runs."),
            (2, "Write <code>Tensor Scaled(Module model, Tensor x, float temperature)</code> that returns the model's output divided by <code>temperature</code> and leaves no temporaries behind.",
             "<code>using var scope = new TensorScope(); using var g = Autograd.NoGrad(); var y = model.Forward(x) * (1f / temperature); return scope.Keep(y);</code> "
             "The forward intermediates and the unscaled output are freed; only <code>y</code> survives."),
            (2, "Explain why <code>after the loop</code> in Section 4.3 shows 512 KiB cached, not 768 KiB.",
             "The step created three tensors: the 256×256 product (256 KiB), its Tanh (256 KiB) and the scalar mean (4 bytes, "
             "rounded in the display). Two large blocks and one tiny block were returned to the cache."),
            (3, "Build a small class <code>Scorer : IDisposable</code> that loads a model once, keeps it in evaluation mode, and scores batches from several threads.",
             "Load and call <code>Eval()</code> in the constructor; expose <code>float[,] Score(float[,] rows) =&gt; model.Predict(rows)</code>; "
             "dispose the model in <code>Dispose()</code>. Because the model is already in evaluation mode, concurrent "
             "<code>Predict</code> calls do not flip its mode; " + ch("webapi") + " adds request limiting."),
        ], PART),
        footer("TensorScope", "Dispose", "Memory cache", "Reference counting", "Finalizer", "InUse", "Reserved",
               "View", "Evaluation mode", "Memory leak"),
    )
