"""Chapter 1 — Getting Started: your first network on CPU and GPU (sample chapter)."""
from gen import *

PART = "I"


def network_svg():
    w, h = 440, 178
    inputs = [(70, 60, "a"), (70, 118, "b")]
    hidden = [(220, 18 + i * 20.5) for i in range(8)]
    out = (370, 89)
    parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for (x1, y1, _) in inputs:
        for (x2, y2) in hidden:
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#b7c3c8" stroke-width="0.7"/>')
    for (x2, y2) in hidden:
        parts.append(f'<line x1="{x2}" y1="{y2}" x2="{out[0]}" y2="{out[1]}" stroke="#b7c3c8" stroke-width="0.7"/>')
    for (x, y, label) in inputs:
        parts.append(f'<circle cx="{x}" cy="{y}" r="13" fill="#e6f2ef" stroke="#0f6b5c" stroke-width="1.4"/>')
        parts.append(svg_text(x, y + 3.5, label, 10, "#0f6b5c"))
    for (x, y) in hidden:
        parts.append(f'<circle cx="{x}" cy="{y}" r="7.5" fill="#ffffff" stroke="#0f6b5c" stroke-width="1.2"/>')
    parts.append(f'<circle cx="{out[0]}" cy="{out[1]}" r="13" fill="#0f6b5c"/>')
    parts.append(svg_text(out[0], out[1] + 3.5, "y", 10, "#ffffff"))
    parts.append(svg_text(70, 172, "input [2]", 8.5, "#56606a"))
    parts.append(svg_text(220, 176, "Linear(2→8) + Tanh", 8.5, "#56606a"))
    parts.append(svg_text(370, 128, "Linear(8→1)", 8.5, "#56606a"))
    parts.append(svg_text(370, 140, "+ Sigmoid", 8.5, "#56606a"))
    parts.append("</svg>")
    return "".join(parts)


def build():
    CURRENT["chapter"] = "1"
    CURRENT["part"] = "I"
    return page(
        topbar("PART I · THE LIBRARY", "CHAPTER 1"),
        h1("CHAPTER 1", "Getting Started: Your First Network on CPU and GPU"),
        brief("NeuralSharp is a neural-network library written entirely in C# for .NET 10. It has no TensorFlow, "
              "no native DLLs of its own and no NuGet dependencies: the CPU backend uses SIMD instructions and all "
              "cores, and the GPU backend talks straight to the NVIDIA display driver. This chapter installs it, "
              "shows how to pick the CPU or the GPU, and trains a first network, the classic XOR problem, both ways."),
        glance(
            "A <b>tensor</b> holds numbers on a <b>device</b> (CPU or GPU); a <b>module</b> turns input tensors into output tensors.",
            "Choose the device once with <code>Device.Default</code>, or per layer with <code>device:</code>. Every "
            "example in this book works unchanged on both.",
            "Training is always the same four lines: forward, loss, <code>Backward()</code>, <code>Step()</code>.",
            "Tensors own device memory: create per-step tensors inside a <code>TensorScope</code>.",
            f"Tiny networks such as XOR run faster on the CPU; the GPU wins once each step carries real work ({ch('performance')}).",
        ),
        h2("1.1 What is in the library"),
        para("Everything lives under the <code>NeuralSharp</code> namespace. You only ever need the handful of "
             "sub-namespaces below; the backends are internal and chosen for you by the device."),
        reftable(["Namespace", "What you use it for", "Chapter"], [
            ["<code>NeuralSharp</code>", "<code>Tensor</code>, <code>Device</code>, <code>Autograd</code>, <code>TensorScope</code>, "
             "<code>Losses</code>, <code>ComputeResources</code>, <code>TokenSampler</code>, <code>ComputeGraph</code>", f"{ch('devices',None)}–{ch('autograd',None)}, {ch('losses',None)}, {ch('generation',None)}"],
            ["<code>NeuralSharp.Layers</code>", "<code>Sequential</code>, <code>Linear</code>, activations, <code>Conv2d</code>, "
             "<code>LSTM</code>, attention, normalization, <code>Embedding</code>", f"{ch('modules',None)}–{ch('attention',None)}"],
            ["<code>NeuralSharp.Optimizers</code>", "<code>Sgd</code>, <code>Adam</code>, <code>AdamW</code>, learning-rate schedules", f"{ch('optimizers',None)}–{ch('schedules',None)}"],
            ["<code>NeuralSharp.Data</code>", "<code>Dataset</code>, CSV loading, scalers, <code>DataLoader</code>", ch("data", None)],
            ["<code>NeuralSharp.Training</code>", "<code>Trainer</code>, <code>Metric</code>, <code>RegressionReport</code>", f"{ch('metrics',None)}, {ch('trainer',None)}"],
            ["<code>NeuralSharp.Diagnostics</code>", "telemetry hooks, loggers, JSON Lines, channels", ch("telemetry", None)],
        ], caption="Table 1.1 — The public namespaces"),
        h2("1.2 Setting up"),
        deriv("Setup, step by step", [
            "Install the <b>.NET 10 SDK</b> (<code>dotnet --version</code> should print 10.x).",
            "Get the source: clone the repository (<code>git clone …/NN.git</code>). The library is the project "
            "<code>src/NeuralSharp</code>.",
            "Reference it from your application: <code>dotnet add reference path/to/src/NeuralSharp/NeuralSharp.csproj</code>. "
            "Alternatively build a package with <code>dotnet pack src/NeuralSharp -c Release</code> and install the "
            "<code>.nupkg</code> from a local folder.",
            "For the GPU, install only the normal <b>NVIDIA display driver</b>. The CUDA Toolkit, cuDNN and cuBLAS are "
            "<i>not</i> needed: the library generates its GPU code (PTX) itself and the driver compiles it on first use.",
            "Always build and run with <code>-c Release</code>; Debug builds are several times slower.",
        ]),
        mex("Which devices do I have?",
            "Run this once on any new machine. It never fails: without an NVIDIA driver it simply reports the CPU only.",
            """
            using NeuralSharp;

            Console.WriteLine($"CUDA available: {Device.IsCudaAvailable} ({Device.CudaDeviceCount} GPU(s))");
            Console.WriteLine($"CPU    : {Device.Cpu.Name}");
            if (Device.IsCudaAvailable)
                Console.WriteLine($"GPU    : {Device.Cuda(0).Name}");
            Console.WriteLine($"Default: {Device.Default}");   // cuda:0 when a GPU exists, else cpu
            """,
            out="""
            CUDA available: True (1 GPU(s))
            CPU    : CPU (16 threads, 8-wide SIMD)
            GPU    : NVIDIA GeForce RTX 5050 Laptop GPU (8150 MiB, 20 SMs)
            Default: cuda:0
            """),
        defbox("Device", "<p>A place where tensors live and where their arithmetic runs: <code>Device.Cpu</code> or "
               "<code>Device.Cuda(n)</code> for the n-th NVIDIA GPU. <code>Device.Default</code> is the device used "
               "whenever you do not name one; it starts as the first GPU if one exists, otherwise the CPU, and you "
               "can assign it.</p>"),
        h2("1.3 Three ideas you use in every program"),
        defbox("Tensor", "<p>An n-dimensional array of 32-bit floats stored on one device, for example a batch of "
               "four XOR inputs is a tensor of shape <code>[4, 2]</code>: 4 rows (samples), 2 columns (features). "
               "Tensors support arithmetic with ordinary operators (<code>a + b</code>, <code>x * 2f</code>) and "
               "methods such as <code>MatMul</code>, <code>Tanh</code> and <code>Mean</code>.</p>"),
        defbox("Module", "<p>A building block that maps an input tensor to an output tensor and owns its trainable "
               "<b>parameters</b> (weights). <code>Linear</code>, <code>Tanh</code> and <code>Sequential</code> are "
               "modules; a whole network is a module too, so you call <code>model.Forward(x)</code> on it.</p>"),
        defbox("Autograd", "<p>Automatic differentiation. Every operation on tensors that require gradients is "
               "recorded; <code>loss.Backward()</code> then walks that record backwards and fills each parameter's "
               "<code>Grad</code> with the direction in which the loss grows fastest. The optimizer moves the "
               "parameters the other way.</p>"),
        honestbox("Where the math lives",
                  "<p>This book uses gradients, the chain rule and activation functions as tools, not as topics. "
                  "The glossary entries <b>Gradient</b>, <b>Chain rule</b> and <b>Sigmoid</b> give one-line hints "
                  "and point to the volume and chapter where each is developed properly: the Pre-Calculus volume "
                  "<i>Numbers, Shapes, and Functions</i> for functions and exponentials, and the Activation Functions "
                  "volume for every activation and its derivative.</p>"),
        h2("1.4 The XOR problem"),
        para("XOR (exclusive or) outputs 1 when exactly one of its two inputs is 1. It is the smallest problem a "
             "single linear layer cannot solve, which makes it the traditional first test of a network with a "
             "hidden layer."),
        reftable(["a", "b", "a XOR b"], [["0", "0", "0"], ["0", "1", "1"], ["1", "0", "1"], ["1", "1", "0"]],
                 caption="Table 1.2 — The four XOR cases (the whole training set)"),
        diagram("Figure 1.1 — The network", network_svg(),
                "Two inputs, a hidden layer of 8 units with Tanh, and one output with Sigmoid so the result reads as a probability."),
        honestbox("Why a hidden layer is needed",
                  "<p>No straight line separates the two 1-cases from the two 0-cases, so one <code>Linear</code> "
                  "layer (plus any squashing function) cannot fit XOR. The proof is the sample chapter of the "
                  "Activation Functions volume (<i>XOR impossibility</i>); here we only use the result.</p>"),
        h2("1.5 Worked example: XOR on the CPU"),
        mex("the complete program, CPU version",
            "Create a console app (<code>dotnet new console</code>), reference the library and replace "
            "<code>Program.cs</code> with this. It uses the low-level training loop so every step is visible; "
            f"{ch('trainer')} shows the same thing with the <code>Trainer</code>.",
            """
            using NeuralSharp;
            using NeuralSharp.Layers;
            using NeuralSharp.Optimizers;

            Device.Default = Device.Cpu;                       // (1) choose the device

            using var inputs  = Tensor.From(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
            using var targets = Tensor.From(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

            var random = new Random(42);                       // (2) seeded, reproducible weights
            using var model = new Sequential
            {
                new Linear(2, 8, random: random), new Tanh(),
                new Linear(8, 1, random: random), new Sigmoid(),
            };
            using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);

            for (int epoch = 1; epoch <= 2000; epoch++)
            {
                using var scope = new TensorScope();           // (3) frees this step's tensors
                var loss = Losses.MeanSquaredError(model.Forward(inputs), targets);
                optimizer.ZeroGrad();                          // (4) forward, loss, backward, step
                loss.Backward();
                optimizer.Step();
                if (epoch % 500 == 0) Console.WriteLine($"epoch {epoch}  loss {loss.Item():F6}");
            }

            using var prediction = model.Predict(inputs);      // (5) inference: no gradients
            Console.WriteLine(prediction);
            """,
            out="""
            epoch 500  loss 0.000019
            epoch 1000  loss 0.000006
            epoch 1500  loss 0.000003
            epoch 2000  loss 0.000002
            Tensor(shape=[4, 1], device=cpu)
            [[0.0007]
             [0.9989]
             [0.9985]
             [0.0014]]
            """,
            after="The outputs are close to 0, 1, 1, 0: the network has learned XOR. Rounding at 0.5 gives exactly the truth table."),
        deriv("What one training step does", [
            "<code>model.Forward(inputs)</code> computes the predictions and records each operation in the autograd graph.",
            "<code>Losses.MeanSquaredError</code> turns predictions and targets into one number, the <b>loss</b> "
            "(hint: the average of the squared differences — glossary <b>MSE</b>).",
            "<code>optimizer.ZeroGrad()</code> clears the previous step's gradients; they would otherwise add up.",
            "<code>loss.Backward()</code> computes every parameter's gradient (glossary <b>Chain rule</b>).",
            "<code>optimizer.Step()</code> moves every parameter a little against its gradient (glossary <b>Adam</b>).",
        ]),
        h2("1.6 The same program on the GPU"),
        para("Only line (1) changes. Because <code>Device.Default</code> is read when tensors and layers are "
             "created, setting it first moves the whole program to the GPU:"),
        snippet("""
            Device.Default = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;   // GPU when present
            """, caption="Line (1), GPU version (falls back to the CPU on machines without an NVIDIA GPU)"),
        para("When a program uses both devices, pass the device explicitly instead of relying on the default. The "
             "three ways below are equivalent; pick whichever reads best in your code."),
        snippet("""
            var gpu = Device.Cuda(0);

            // (a) per layer and per tensor
            var layer = new Linear(2, 8, device: gpu);
            var x = Tensor.From(new float[,] { { 0, 1 } }, device: gpu);

            // (b) build on the CPU, then move the whole model (and tensors) once
            using var model = new Sequential { new Linear(2, 8), new Tanh(), new Linear(8, 1) };
            model.To(gpu);
            using var xOnGpu = x.To(gpu);

            // (c) choose for the whole process
            Device.Default = gpu;
            """, caption="Choosing a device explicitly"),
        reftable(["How", "Effect"], [
            ["<code>Device.Default = Device.Cpu</code>", "Everything created afterwards lives on the CPU."],
            ["<code>Device.Default = Device.Cuda(0)</code>", "Everything created afterwards lives on GPU 0 (throws if there is no GPU)."],
            ["<code>device: Device.Cuda(1)</code>", "One layer, tensor or loader on GPU 1."],
            ["<code>model.To(device)</code> / <code>tensor.To(device)</code>", "Moves (copies) an existing model or tensor."],
            ["<code>NEURALSHARP_DISABLE_CUDA=1</code>", "Environment variable: behave as if no GPU existed."],
            ["<code>--cpu</code> / <code>--cuda</code> / <code>--device cuda:1</code>", "Command-line switches of every sample project."],
        ], caption="Table 1.3 — Selecting the CPU or the GPU"),
        trap("mixing devices",
             "<p>An operation between a CPU tensor and a GPU tensor throws <i>Tensors are on different devices</i>. "
             "Create the data on the model's device (or call <code>.To(device)</code>) before calling "
             "<code>Forward</code>.</p>"),
        trap("expecting the GPU to win on XOR",
             "<p>Each GPU operation has a fixed launch cost of a few microseconds. XOR has 33 parameters and 4 "
             "samples, so the CPU finishes first. The GPU pays off with wide layers, large batches, convolutions "
             "and transformers: the GPT of " + ch("gpt") + " trains about 8× faster on a laptop RTX 5050 than on its CPU.</p>"),
        h2("1.7 Saving the model and using it later"),
        mex("train once, predict many times",
            None,
            """
            model.Save("xor.weights");                          // after training

            // later, in another run or program (any device):
            using var restored = new Sequential
            {
                new Linear(2, 8), new Tanh(), new Linear(8, 1), new Sigmoid(),
            };
            restored.Load("xor.weights");                       // same architecture required
            float[,] result = restored.Predict(new float[,] { { 1, 0 } });
            Console.WriteLine(result[0, 0] >= 0.5f ? 1 : 0);    // prints 1
            """,
            after="Weights files are device-independent: train on the GPU, load on a CPU-only server, or the reverse."),
        h2("1.8 Memory: who frees a tensor?"),
        para("A tensor holds device memory, which on a GPU is limited and invisible to the .NET garbage collector. "
             "NeuralSharp recycles memory through a pool, and three habits keep it tidy:"),
        reftable(["Situation", "What to do"], [
            ["Long-lived tensors (datasets, inputs you reuse)", "<code>using var t = Tensor.From(...)</code>"],
            ["Everything created inside one training step", "<code>using var scope = new TensorScope();</code> at the top of the loop body"],
            ["Model parameters, gradients, optimizer state", "Nothing: they are never captured by a scope; dispose the model/optimizer when done"],
            ["Inference", "<code>model.Predict(x)</code> frees its intermediates itself; dispose only the result"],
        ], caption="Table 1.4 — Memory habits"),
        trap("disposing a tensor that the graph still needs",
             "<p>Do not put <code>using</code> on intermediate results such as the prediction or the loss before "
             "<code>Backward()</code> has run: the backward pass reads them. A <code>TensorScope</code> releases "
             "them at the right moment, after the step.</p>"),
        trap("forgetting ZeroGrad",
             "<p>Gradients accumulate across <code>Backward()</code> calls by design. Without "
             "<code>optimizer.ZeroGrad()</code> every step would use the sum of all previous gradients and "
             "training diverges.</p>"),
        practice([
            (1, "Change the XOR program so it runs on the GPU when one is present and prints which device it used.",
             "Set <code>Device.Default = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;</code> as the first line "
             "and print <code>Device.Default</code> and <code>Device.Default.Name</code>."),
            (1, "Replace the hidden layer's <code>Tanh</code> with <code>ReLU</code> and train again. Does it still learn XOR?",
             "Usually yes with 8 hidden units, but a few seeds get stuck because ReLU units can stop learning "
             "(see glossary <b>ReLU</b>); Tanh is the safer choice for this tiny problem."),
            (2, "Shrink the hidden layer to 2 units. Train with several seeds and report how often it succeeds.",
             "Two hidden units are enough in principle, but training fails for a noticeable fraction of seeds; "
             "wider hidden layers make the optimization much more reliable."),
            (2, "Save the trained model, then write a second program that loads it on the CPU and prints the four predictions.",
             "Build the identical <code>Sequential</code>, call <code>Load(path)</code>, then "
             "<code>Predict(new float[,] {{0,0},{0,1},{1,0},{1,1}})</code>."),
            (3, "Time 2,000 training epochs on the CPU and on the GPU, then repeat with a hidden layer of 4,096 units and a batch of 4,096 random XOR samples. Explain the difference.",
             "The tiny network is faster on the CPU (launch overhead dominates); the wide one is faster on the GPU "
             "because each kernel now does enough arithmetic to hide the launch cost."),
        ], PART),
        footer("Tensor", "Device", "Module", "Autograd", "Parameter", "Loss", "MSE", "Epoch", "Optimizer", "Adam",
               "XOR", "Tanh", "Sigmoid", "ReLU", "Gradient", "Chain rule", "TensorScope"),
    )

