"""Chapter 24 — Testing, Gradient Checking and Debugging."""
from gen import *

PART = "IV"


def build():
    return page(
        chapter_open(
            "testing",
            "A neural network that is wrong usually still runs: it just learns badly. This chapter shows how the "
            "library tests itself on every device, how to test your own layers and models the same way, and a "
            "systematic procedure for the classic failures: loss not falling, NaN, shape errors, memory growth, and "
            "CPU and GPU disagreeing.",
            "The test project runs every test on every available device: <code>dotnet run -c Release --project tests/NeuralSharp.Tests</code>.",
            "Three kinds of test: compare with a reference implementation, check gradients with finite differences, and check that a model can learn.",
            "Overfit one tiny batch first: a model that cannot memorize 8 samples has a bug.",
            "Most silent bugs are data bugs: unscaled inputs, wrong labels, leakage between train and validation.",
            "Compare CPU and GPU with a relative tolerance, never bit for bit.",
        ),
        h2("24.1 The library's own tests"),
        para("The test project is a self-contained console program (no test framework needed). It runs each test once on "
             "the CPU and once on every GPU, inside a <code>TensorScope</code>, and prints a line per test:"),
        output("""
            CUDA not available (the NVIDIA driver library (libcuda.so.1 / libcuda.so) was not found); testing the CPU only.
            == cpu: CPU (4 threads, 8-wide SIMD)
              PASS matmul matches reference (all transposes, beta 0 and 1) (87 ms)
              PASS element-wise ops match reference (25 ms)
              PASS large tensors (parallel / multi-block paths) (235 ms)
              PASS gradient: sigmoid, sum (7 ms)
              ...
              PASS batched cached decoding matches sequences decoded one by one (5 ms)
              PASS graph replay gives the same tokens as direct execution (12 ms)
              PASS sampler: distribution, top-k, temperature, determinism, CPU parity (19 ms)
            56 passed, 0 failed
            """, caption="The test run in this book's container (CPU only); on a machine with one GPU, 112 tests run"),
        reftable(["Test kind", "Examples in the suite", "Catches"], [
            ["Reference comparison", "matmul (all transposes), element-wise ops, softmax, convolution, pooling against simple loops", "Wrong kernels, indexing errors"],
            ["Gradient checks", "every operation and every layer, including dropout with a fixed mask", "Wrong backward formulas"],
            ["Learning tests", "optimizers minimize a quadratic; XOR and small tasks train", "Wiring errors that pass the other tests"],
            ["Property tests", "save/load round-trip, scopes free memory, cached decoding equals full decoding, graph replay equals direct execution", "Lifetime and consistency errors"],
            ["Device parity", "the same tests on CPU and GPU; the sampler picks the same tokens", "Backend-specific bugs"],
        ], caption="Table 24.1 — What the suite tests"),
        h2("24.2 Testing your own layers"),
        snippet("""
            static void GradCheck(Device device, int[] shape, Func<Tensor, Tensor> f, float tolerance = 2e-2f)
            {
                var random = new Random(1);
                var x0 = new float[shape.Aggregate(1, (a, b) => a * b)];
                for (int i = 0; i < x0.Length; i++) x0[i] = random.NextSingle() * 2 - 1;

                using var scope = new TensorScope();
                var x = Tensor.From(x0, shape, device, requiresGrad: true);
                f(x).Backward();
                float[] analytic = x.Grad!.ToArray();

                const float h = 1e-2f;
                using var noGrad = Autograd.NoGrad();
                for (int i = 0; i < x0.Length; i++)
                {
                    var plus = (float[])x0.Clone();  plus[i] += h;
                    var minus = (float[])x0.Clone(); minus[i] -= h;
                    float numeric = (f(Tensor.From(plus, shape, device)).Item()
                                   - f(Tensor.From(minus, shape, device)).Item()) / (2 * h);
                    if (MathF.Abs(numeric - analytic[i]) > tolerance * Math.Max(1f, MathF.Abs(numeric)))
                        throw new Exception($"element {i}: autograd {analytic[i]}, numeric {numeric}");
                }
            }

            // a layer's input gradient, on every device
            foreach (var device in new[] { Device.Cpu }.Concat(Device.IsCudaAvailable ? [Device.Cuda()] : []))
            {
                using var layer = new FeatureGate(4, device, new Random(3));        // your custom layer (Chapter 6)
                GradCheck(device, [3, 4], x => layer.Forward(x).Tanh().Sum());
                Console.WriteLine($"{device}: gradient OK");
            }
            """.replace("(Chapter 6)", f"({ch('modules')})"), caption="A reusable gradient check (adapted from the test suite)"),
        para("For a layer's <i>parameter</i> gradients, perturb each parameter value instead of the input: read "
             "the parameter with <code>ToArray()</code>, change one value, write it back by loading a file or by "
             "building the layer from known values, and compare the change in loss with the parameter's "
             "<code>Grad</code>. Checking the input gradient, as above, already exercises the whole backward pass "
             "of the layer."),
        cpugpu("comparing devices",
               """
               using var cpuModel = Factory.Create(Device.Cpu);
               cpuModel.Save("tmp.weights");
               float[,] cpuOut = cpuModel.Predict(batch);
               """,
               """
               using var gpuModel = Factory.Create(Device.Cuda());
               gpuModel.Load("tmp.weights");                        // identical weights
               float[,] gpuOut = gpuModel.Predict(batch);
               // compare element by element: |a - b| <= 1e-4 * max(1, |a|)
               """),
        h2("24.3 When training goes wrong"),
        deriv("A procedure that finds most bugs", [
            "<b>Look at the data.</b> Print a few rows of features and targets after scaling. Check shapes, ranges, "
            "NaN values and that labels match inputs.",
            "<b>Check the starting loss.</b> Cross-entropy should start near ln K (" + ch("losses") + "); MSE on standardized "
            "targets near 1.",
            "<b>Overfit a tiny batch.</b> Train on 8 samples for a few hundred steps without dropout or weight decay. The loss "
            "must go almost to zero. If not, the model, loss or optimizer wiring is wrong.",
            "<b>Check gradients</b> of any custom layer or loss (Section 24.2).",
            "<b>Scale up</b> to the full data; then tune the learning rate (" + ch("optimizers") + ") and capacity.",
            "<b>Watch validation</b> for overfitting and use early stopping (" + ch("trainer") + ").",
        ]),
        reftable(["Symptom", "Likely causes", "Where to look"], [
            ["Loss does not fall at all", "Learning rate far too low or too high; <code>ZeroGrad</code>/<code>Step</code> missing; parameters not given to the optimizer; frozen by mistake", ch("optimizers") + ", " + ch("autograd")],
            ["Loss becomes NaN", "Rate too high; unscaled inputs; <code>Log</code> of 0 or negative numbers in a custom loss; NaN in the data", ch("schedules") + " (clipping), " + ch("data")],
            ["Loss stuck at ln K (classification)", "Labels shuffled relative to inputs; wrong target format; Softmax before cross-entropy", ch("losses")],
            ["Training good, validation bad", "Overfitting; leakage-free split? scaler fitted on training data only?", ch("norm") + ", " + ch("data")],
            ["Validation suspiciously good", "Leakage: duplicates or overlapping windows in both sets", ch("trainer")],
            ["Predictions all the same value", "Dead ReLUs; too high a rate; targets not scaled; model in the wrong mode", ch("dense")],
            ["Different result on every <code>Predict</code>", "Model in training mode (dropout on)", ch("modules")],
            ["Shape exception", "Read the message: it names both shapes; trace shapes layer by layer", ch("conv") + " (Section 10.3)"],
            ["Memory grows every step", "Missing <code>TensorScope</code>; tensors stored across steps", ch("memory")],
            ["CPU and GPU differ slightly", "Normal float32 rounding differences", ch("cuda")],
            ["CPU and GPU differ a lot", "Different weights (seed, loading), or a device bug: report it with a minimal repro", "Section 24.2"],
        ], caption="Table 24.2 — Symptoms and causes"),
        mex("overfitting one tiny batch as a smoke test", None,
            """
            // 8 random samples, random targets: a working model must memorize them.
            var r = new Random(0);
            using var x = Tensor.Uniform([8, 10], -1, 1, r);
            using var y = Tensor.Uniform([8, 1], -1, 1, r);
            using var model = new Sequential { new Linear(10, 64, random: r), new ReLU(), new Linear(64, 1, random: r) };
            using var opt = new Adam(model.Parameters(), 1e-2f);
            float loss = 0;
            for (int step = 0; step < 500; step++)
            {
                using var scope = new TensorScope();
                var l = Losses.MeanSquaredError(model.Forward(x), y);
                opt.ZeroGrad(); l.Backward(); opt.Step();
                loss = l.Item();
            }
            Console.WriteLine(loss < 1e-4 ? $"OK: memorized ({loss:E1})" : $"PROBLEM: loss {loss}");
            """,
            out="""
            OK: memorized (6.9E-018)
            """),
        h2("24.4 Tools for looking inside"),
        reftable(["Tool", "Shows"], [
            ["<code>model.Summary()</code>", "Structure and parameter counts (" + ch("modules") + ")"],
            ["Shape trace (<code>foreach (var layer in model)</code>)", "Every intermediate shape (" + ch("conv") + ")"],
            ["<code>ConsoleLogger(TelemetryLevel.Layers)</code>", "Each layer's input/output shapes and time"],
            ["<code>ConsoleLogger(TelemetryLevel.Operations)</code>", "Every operation, forward and backward (" + ch("telemetry") + ")"],
            ["<code>TelemetryLevel.Gradients</code>", "Gradient norm per batch: exploding or vanishing gradients"],
            ["<code>tensor.ToString()</code>", "Values of small tensors"],
            ["<code>ComputeResources.GetMemoryUsage(device)</code>", "Leaks (" + ch("memory") + ")"],
            ["<code>NEURALSHARP_DISABLE_CUDA=1</code>", "Run a GPU program on the CPU to compare behaviour"],
        ], caption="Table 24.3 — Debugging tools"),
        practice([
            (1, "A new 5-class classifier starts with a loss of 12.4. What do you check first?",
             "The starting loss should be near ln 5 ≈ 1.61. Check input scaling (huge activations give huge logits), the "
             "target format, and that there is no Softmax before the loss."),
            (1, "Why overfit a tiny batch before training on everything?",
             "It separates bugs from tuning: any correctly wired model can memorize 8 samples. If it cannot, something is "
             "structurally wrong, and no amount of tuning on the full data will fix it."),
            (2, "Write a test that a model gives the same output before saving and after loading into a fresh instance.",
             "Build with a factory, predict a fixed batch, <code>Save</code>, create a second instance, <code>Load</code>, "
             "predict again, and compare with a tolerance of about 1e-6 (the same device gives identical values)."),
            (2, "Run the gradient check of Section 24.2 on <code>Losses.BinaryCrossEntropyWithLogits</code> with fixed targets.",
             "<code>GradCheck(device, [6, 1], z =&gt; Losses.BinaryCrossEntropyWithLogits(z, targets))</code> with "
             "<code>targets</code> a fixed tensor of 0s and 1s created on the same device; it passes."),
            (3, "Design a regression test for a trained production model.",
             "Keep a small fixed input set with the expected outputs saved at release time; the test loads the model and "
             "scalers, predicts, and checks every output within a tolerance, and also checks a metric (e.g. MAE on a held-out "
             "set) against a threshold. Run it on both devices in CI where a GPU is available."),
        ], PART),
        footer("Unit test", "Reference implementation", "Finite difference", "Smoke test", "Overfitting a batch",
               "NaN", "Device parity", "Data leakage"),
    )
