"""Chapter 6 — Modules, Sequential and Model Files."""
from gen import *

PART = "II"


def build():
    return page(
        chapter_open(
            "modules",
            "Every layer, every activation and every whole network in NeuralSharp is a <code>Module</code>. This "
            "chapter explains what a module is and does, how <code>Sequential</code> chains modules, how to inspect "
            "a model, how to write your own layers, and how model files are saved, loaded and moved between devices.",
            "A module maps an input tensor to an output tensor with <code>Forward</code> and owns its <b>parameters</b> (and non-trainable <b>buffers</b>).",
            "<code>Sequential</code> runs modules in order and is built with a collection initializer.",
            "<code>Summary()</code>, <code>ParameterCount</code>, <code>Parameters()</code> and indexing show what a model contains.",
            "Custom layers: derive from <code>Module</code>, implement <code>ForwardCore</code>, and list children or parameters.",
            "<code>Save</code>/<code>Load</code> write and read raw parameter values; the code that builds the model must be the same in both programs.",
        ),
        h2("6.1 What a module is"),
        reftable(["Member", "Purpose"], [
            ["<code>Forward(Tensor input)</code>", "Computes the output; records gradients when autograd is on. Publishes telemetry when enabled."],
            ["<code>Predict(Tensor)</code> / <code>Predict(float[,])</code>", "Inference: evaluation mode, no gradients, intermediates freed (" + ch("inference") + ")."],
            ["<code>Parameters()</code>", "The trainable tensors of the module and all its children, in a stable order."],
            ["<code>Buffers()</code>", "Non-trainable state (e.g. BatchNorm running statistics); saved and moved, not optimized."],
            ["<code>Children()</code>", "Direct sub-modules (empty for leaf layers)."],
            ["<code>Train()</code> / <code>Eval()</code> / <code>IsTraining</code>", "Training or evaluation mode, applied to all children."],
            ["<code>To(device)</code>", "Moves parameters and buffers; returns the module."],
            ["<code>Save(path)</code> / <code>Load(path)</code>", "Writes / reads all parameter and buffer values."],
            ["<code>Summary()</code>, <code>ParameterCount</code>", "Readable structure, number of trainable values."],
            ["<code>Name</code>, <code>DisplayName</code>", "Optional name used in summaries and telemetry."],
            ["<code>Dispose()</code>", "Frees the parameters' memory."],
        ], caption="Table 6.1 — The Module API"),
        defbox("Parameter and buffer",
               "<p>A <b>parameter</b> is a tensor the optimizer changes: weights and biases, created with "
               "<code>requiresGrad</code> on and never captured by a <code>TensorScope</code>. A <b>buffer</b> is state "
               "the layer updates itself or keeps fixed, such as running averages or a causal mask; it is saved and "
               "moved with the model but has no gradient.</p>"),
        h2("6.2 Sequential"),
        para("<code>Sequential</code> feeds each module's output into the next. It supports collection-initializer "
             "syntax, <code>Add</code>, indexing and <code>foreach</code>, and it is itself a module, so sequences nest."),
        mex("building and inspecting a model", None,
            """
            using NeuralSharp;
            using NeuralSharp.Layers;

            var random = new Random(1);
            using var model = new Sequential
            {
                new Linear(4, 16, random: random), new ReLU(),
                new Linear(16, 16, random: random), new ReLU(),
                new Linear(16, 3, random: random),
            };
            model.Name = "iris-mlp";

            Console.WriteLine(model.Summary());
            Console.WriteLine($"{model.Count} layers; first: {model[0]}; params {model.ParameterCount}");
            foreach (var p in model.Parameters())
                Console.Write($"[{string.Join(",", p.Shape.ToArray())}] ");
            """,
            out="""
            iris-mlp
              Linear(4 -> 16)  [80 params]
              ReLU
              Linear(16 -> 16)  [272 params]
              ReLU
              Linear(16 -> 3)  [51 params]
            Total trainable parameters: 403
            5 layers; first: Linear(4 -> 16); params 403
            [4,16] [16] [16,16] [16] [16,3] [3]
            """,
            after="A <code>Linear(in, out)</code> has in·out weights plus out biases: 4·16 + 16 = 80."),
        cpugpu("building the model on a device",
               """
               Device.Default = Device.Cpu;
               using var model = BuildModel();      // layers are created on Device.Default
               """,
               """
               using var model = BuildModel();      // built on the CPU (or wherever) ...
               model.To(Device.Cuda());             // ... then moved in one call
               """,
               "Either set the device before building, or pass <code>device:</code> to each layer, or build and call "
               "<code>To</code>. Moving copies each parameter once and frees the original."),
        h2("6.3 Lambda: any tensor function as a layer"),
        para("For a step that has no parameters, such as averaging over time or reshaping, wrap a function in "
             "<code>Lambda</code>. It is differentiated automatically like everything else."),
        mex("averaging a sequence inside a Sequential", None,
            """
            using var pool = new Sequential
            {
                new Linear(3, 5),                                // [N, T, 3] -> [N, T, 5]
                new Lambda(x => x.Mean(1), "MeanOverTime"),      // [N, T, 5] -> [N, 5]
            };
            using var scope = new TensorScope();
            Console.WriteLine(string.Join(",", pool.Forward(Tensor.Ones([2, 7, 3])).Shape.ToArray()));
            Console.WriteLine(pool.Summary());
            """,
            out="""
            2,5
            Sequential(2 layers)
              Linear(3 -> 5)  [20 params]
              MeanOverTime
            Total trainable parameters: 20
            """),
        h2("6.4 Writing your own layers"),
        para("There are two kinds of custom module. A <b>composite</b> module is built from other modules and lists "
             "them in <code>Children()</code>; everything else (parameters, saving, moving, modes) then works "
             "automatically. A <b>leaf</b> module owns tensors directly: it creates them with "
             "<code>CreateParameter</code>, returns them from <code>Parameters()</code>, and moves them in "
             "<code>MoveTo</code>."),
        mex("a composite module: a residual block",
            "A residual block adds its input to the output of a small sub-network (glossary <b>Residual connection</b>). "
            "It is the building block of deep networks, and transformers use it twice per layer.",
            """
            sealed class Residual : Module
            {
                private readonly Linear _a, _b;

                public Residual(int width, Random? random = null)
                {
                    _a = new Linear(width, width, random: random);
                    _b = new Linear(width, width, random: random);
                }

                protected override Tensor ForwardCore(Tensor x) => x + _b.Forward(_a.Forward(x).Relu());
                public override IEnumerable<Module> Children() => [_a, _b];
                public override string ToString() => "Residual";
            }

            using var net = new Sequential { new Linear(8, 8), new Residual(8), new ReLU() };
            Console.WriteLine(net.Summary());
            """,
            out="""
            Sequential(3 layers)
              Linear(8 -> 8)  [72 params]
              Residual
                Linear(8 -> 8)  [72 params]
                Linear(8 -> 8)  [72 params]
              ReLU
            Total trainable parameters: 216
            """),
        mex("a leaf module with its own parameter",
            "A learned gate: every feature is multiplied by a value between 0 and 1 computed from all features. "
            "<code>CreateParameter</code>, <code>UniformValues</code> and <code>MoveTensor</code> are protected helpers "
            "of <code>Module</code>.",
            """
            /// y = x * sigmoid(x · W)
            sealed class FeatureGate : Module
            {
                public FeatureGate(int features, Device? device = null, Random? random = null) =>
                    Weight = CreateParameter(
                        UniformValues(features * features, 1f / MathF.Sqrt(features), random ?? Random.Shared),
                        [features, features], device ?? Device.Default);

                public Tensor Weight { get; private set; }

                protected override Tensor ForwardCore(Tensor x) => x * x.MatMul(Weight).Sigmoid();
                public override IEnumerable<Tensor> Parameters() => [Weight];
                protected override void MoveTo(Device device) => Weight = MoveTensor(Weight, device);
                public override string ToString() => $"FeatureGate({Weight.Shape[0]})";
            }
            """,
            after="A <code>Sequential { new FeatureGate(3), new Linear(3, 1) }</code> reports 9 + 4 = 13 parameters, "
                  "trains with any optimizer and saves with <code>Save</code>."),
        deriv("Checklist for a custom layer", [
            "Derive from <code>Module</code> and override <code>ForwardCore</code> using tensor operations only; autograd "
            "differentiates it for you.",
            "Composite: return sub-modules from <code>Children()</code>. Leaf: create tensors with <code>CreateParameter</code> "
            "(trainable) or <code>CreateBuffer</code> (state), return them from <code>Parameters()</code> / <code>Buffers()</code>, "
            "and reassign them in <code>MoveTo</code> with <code>MoveTensor</code>.",
            "Take <code>Device? device = null</code> and <code>Random? random = null</code> constructor arguments like the "
            "built-in layers, so callers control placement and reproducibility.",
            "Use <code>IsTraining</code> if the layer behaves differently in training and evaluation.",
            "Override <code>ToString()</code> for readable summaries, and check the gradients numerically once (" + ch("autograd") + ").",
        ]),
        honestbox("Broadcasting inside custom layers",
                  "<p>Only <code>+</code> broadcasts (" + ch("tensors") + "). To multiply <code>[N, F]</code> by a "
                  "per-feature parameter <code>s</code> of shape <code>[F]</code>, first expand <code>s</code> to "
                  "<code>[N, F]</code> with a matrix product, which keeps the gradient flowing to <code>s</code>: "
                  "<code>x * Tensor.Ones([N, 1], x.Device).MatMul(s.Reshape(1, -1))</code>.</p>"),
        h2("6.5 Model files"),
        para("<code>model.Save(path)</code> writes every parameter and buffer, in the order of <code>Parameters()</code> "
             "followed by <code>Buffers()</code>, as raw float32 values with their shapes. It does <b>not</b> write the "
             "architecture: the loading program must build the identical model first and then call "
             "<code>Load</code>, which checks the count and every shape."),
        reftable(["Bytes", "Content"], [
            ["4", "Magic number identifying a NeuralSharp weights file"],
            ["4", "Number of tensors"],
            ["per tensor: 4 + 4·rank", "Rank, then each dimension"],
            ["per tensor: 4·size", "The values, float32 little-endian (converted on big-endian machines)"],
        ], caption="Table 6.2 — The weights file format"),
        para("For the 403-parameter model above the file is 1,680 bytes: 8 header bytes, 60 bytes of shapes and "
             "1,612 bytes of values. Files are independent of the device: a model trained on the GPU loads on a "
             "CPU-only machine and vice versa."),
        mex("what Load reports when the architecture differs", None,
            """
            model.Save("iris.weights");

            using var wider = new Sequential
            {
                new Linear(4, 32), new ReLU(), new Linear(32, 32), new ReLU(), new Linear(32, 3),
            };
            try { wider.Load("iris.weights"); }
            catch (InvalidDataException e) { Console.WriteLine(e.Message); }

            using var smaller = new Sequential { new Linear(4, 3) };
            try { smaller.Load("iris.weights"); }
            catch (InvalidDataException e) { Console.WriteLine(e.Message); }
            """,
            out="""
            Shape mismatch: file has [4, 16], model has [4, 32].
            The file has 6 parameter tensors but the model has 2.
            """),
        snippet("""
            // One factory used by BOTH the training program and every program that loads the model.
            public static class IrisModel
            {
                public const int Features = 4, Classes = 3;

                public static Sequential Create(Device? device = null, Random? random = null) => new()
                {
                    new Linear(Features, 16, device: device, random: random), new ReLU(),
                    new Linear(16, 16, device: device, random: random), new ReLU(),
                    new Linear(16, Classes, device: device, random: random),
                };
            }

            // training:  using var model = IrisModel.Create(); ... model.Save("iris.weights");
            // serving:   using var model = IrisModel.Create(Device.Cpu); model.Load("iris.weights"); model.Eval();
            """, caption="The pattern that keeps training and inference in step"),
        trap("changing the architecture after saving",
             "<p>Adding a layer, changing a width or reordering layers makes old files unloadable (the error names the "
             "first mismatch). Version your files (<code>iris-v2.weights</code>) or keep the old factory. Layers "
             "without parameters (activations, <code>Dropout</code>, <code>Flatten</code>, <code>Lambda</code>) "
             "do not appear in the file, so adding or removing them does not break loading, but it changes what "
             "the model computes.</p>"),
        h2("6.6 Training and evaluation mode"),
        para("<code>Train()</code> and <code>Eval()</code> set <code>IsTraining</code> on the module and all its "
             "children. Two built-in layers care: <code>Dropout</code> is off in evaluation mode, and "
             "<code>BatchNorm</code> uses its running statistics instead of the current batch (" + ch("norm") + "). "
             "<code>Predict</code> switches to evaluation for the call and restores the previous mode afterwards."),
        reftable(["When", "Mode"], [
            ["Training steps", "<code>Train()</code> (the default for a new module)"],
            ["Validation during training", "<code>Eval()</code> for the evaluation, then <code>Train()</code>; the Trainer does this for you"],
            ["After loading for inference", "<code>Eval()</code> once, then keep it"],
            ["Monte-Carlo dropout (uncertainty estimates)", "<code>Train()</code> during inference, under <code>Autograd.NoGrad()</code>"],
        ], caption="Table 6.3 — Which mode when"),
        practice([
            (1, "How many parameters does <code>Sequential { new Linear(10, 64), new ReLU(), new Linear(64, 64), new ReLU(), new Linear(64, 1) }</code> have? Check with <code>ParameterCount</code>.",
             "(10·64 + 64) + (64·64 + 64) + (64·1 + 1) = 704 + 4,160 + 65 = 4,929."),
            (1, "Print the output shape after each layer of a model, as a debugging aid.",
             "Loop over the <code>Sequential</code>: <code>foreach (var layer in model) { x = layer.Forward(x); Console.WriteLine($\"{layer} -&gt; [{string.Join(\", \", x.Shape.ToArray())}]\"); }</code> "
             "inside a <code>TensorScope</code> (" + ch("conv") + " uses exactly this)."),
            (2, "Write a composite <code>MlpBlock(int width, float dropout)</code> = Linear → ReLU → Dropout, and build a network from three of them.",
             "Hold a <code>Linear</code> and a <code>Dropout</code>, return both from <code>Children()</code>, and in "
             "<code>ForwardCore</code> compute <code>_drop.Forward(_linear.Forward(x).Relu())</code>. Train/Eval then "
             "reach the Dropout automatically."),
            (2, "Save a model trained on the GPU and load it on the CPU in a separate program.",
             "Train with <code>Device.Default = Device.Cuda()</code> and <code>model.Save(path)</code>. In the second program "
             "create the model with <code>device: Device.Cpu</code> (or <code>Device.Default = Device.Cpu</code>) using the same "
             "factory, then <code>Load(path)</code>."),
            (3, "Write a leaf layer <code>Scale(int features)</code> that multiplies each feature by its own learned factor (initially 1).",
             "Create <code>S = CreateParameter(Enumerable.Repeat(1f, features).ToArray(), [features], device)</code>; in "
             "<code>ForwardCore</code> return <code>x * Tensor.Ones([x.Shape[0], 1], x.Device).MatMul(S.Reshape(1, -1))</code>; "
             "return <code>[S]</code> from <code>Parameters()</code> and move it in <code>MoveTo</code>."),
        ], PART),
        footer("Module", "Parameter", "Buffer", "Sequential", "Lambda", "Composite module", "Residual connection",
               "Weights file", "Training mode", "Evaluation mode", "Factory method"),
    )
