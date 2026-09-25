"""Chapter 7 — Dense Layers and Activations."""
from gen import *

PART = "II"


def curves_svg():
    import math
    w, h = 470, 170
    x0, y0, sx, sy = 40, 90, 22, 55   # origin and scale: x in [-4, 4] -> 176 px each side? keep 4 panels
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    funcs = [
        ("Sigmoid", lambda x: 1 / (1 + math.exp(-x))),
        ("Tanh", math.tanh),
        ("ReLU", lambda x: max(x, 0.0)),
        ("GELU", lambda x: 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))),
    ]
    for i, (name, fn) in enumerate(funcs):
        ox = 20 + i * 115
        cx, cy = ox + 50, 95
        p.append(f'<rect x="{ox}" y="20" width="100" height="130" rx="4" fill="#fbfdfc" stroke="#d5dcdf"/>')
        p.append(f'<line x1="{ox + 4}" y1="{cy}" x2="{ox + 96}" y2="{cy}" stroke="#b7c3c8" stroke-width="0.8"/>')
        p.append(f'<line x1="{cx}" y1="26" x2="{cx}" y2="146" stroke="#b7c3c8" stroke-width="0.8"/>')
        pts = []
        for k in range(81):
            x = -3 + 6 * k / 80
            y = max(-1.2, min(fn(x), 2.2))
            pts.append(f"{cx + x * 15:.1f},{cy - y * 22:.1f}")
        p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#0f6b5c" stroke-width="1.8"/>')
        p.append(svg_text(cx, 14, name, 9, "#0f6b5c"))
        p.append(svg_text(ox + 94, cy + 11, "x", 7.4, "#56606a", "end"))
    p.append(svg_text(235, 166, "each panel: x from −3 to 3; the vertical line is x = 0, the horizontal line y = 0", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "dense",
            "The dense (fully connected) layer is the workhorse of neural networks: every output is a weighted sum "
            "of every input plus a bias. Between dense layers sit activation functions, which make the network more "
            "than one big linear map. This chapter covers <code>Linear</code>, the five activations, how weights are "
            "initialized, and how to size and stack dense layers into a multilayer perceptron (MLP).",
            "<code>Linear(in, out)</code> computes <code>x · W + b</code> on <code>[..., in]</code> → <code>[..., out]</code>, for any number of leading dimensions.",
            "Activations: <code>ReLU</code> (default for hidden layers), <code>GELU</code> (transformers), <code>Tanh</code> (small nets, recurrent), <code>Sigmoid</code> (yes/no outputs), <code>Softmax</code> (class probabilities at inference).",
            "Put an activation after every hidden <code>Linear</code>, and none (raw scores) after the last one when the loss applies its own.",
            "Weights start Xavier-uniform, biases at zero; pass a seeded <code>Random</code> to make runs repeatable.",
            "Start with 2 hidden layers of 32–256 units; widen before you deepen.",
        ),
        h2("7.1 Linear"),
        reftable(["Constructor argument", "Default", "Meaning"], [
            ["<code>inFeatures</code>", "—", "Size of the last input dimension"],
            ["<code>outFeatures</code>", "—", "Size of the last output dimension"],
            ["<code>bias</code>", "<code>true</code>", "Learn an additive bias per output"],
            ["<code>device</code>", "<code>Device.Default</code>", "Where the weight and bias live"],
            ["<code>random</code>", "<code>Random.Shared</code>", "Source of the initial weights"],
        ], caption="Table 7.1 — new Linear(inFeatures, outFeatures, bias, device, random)"),
        mex("one layer, three kinds of input", None,
            """
            using var layer = new Linear(8, 4);
            using var scope = new TensorScope();
            Console.WriteLine(string.Join(",", layer.Forward(Tensor.Ones([32, 8])).Shape.ToArray()));      // a batch
            Console.WriteLine(string.Join(",", layer.Forward(Tensor.Ones([32, 10, 8])).Shape.ToArray()));  // sequences
            Console.WriteLine(new Linear(8, 4, bias: false));
            """,
            out="""
            32,4
            32,10,4
            Linear(8 -> 4, no bias)
            """,
            after="On sequences the same weights are applied at every time step, which is how transformers use their "
                  "feed-forward layers. <code>layer.Weight</code> is <code>[8, 4]</code> and <code>layer.Bias</code> "
                  "is <code>[4]</code>; both are ordinary tensors you can read with <code>ToArray()</code>."),
        defbox("Xavier (Glorot) initialization",
               "<p><code>Linear</code> draws its weights uniformly from ±√(6 / (in + out)) so that the size of the "
               "signal stays roughly constant from layer to layer at the start of training (glossary <b>Initialization</b>). "
               "<code>Conv2d</code> uses the He variant, ±√(6 / fanIn), which suits ReLU. You rarely need to change "
               "this; you do need to seed it when you want identical runs.</p>"),
        h2("7.2 The activations"),
        diagram("Figure 7.1 — The four element-wise activations", curves_svg(),
                "Sigmoid and Tanh saturate (flatten) for large |x|; ReLU and GELU keep growing for positive x."),
        reftable(["Layer", "Output range", "Choose it for", "Watch out for"], [
            ["<code>ReLU</code>", "[0, ∞)", "Hidden layers of MLPs and CNNs; fast, the default", "Units stuck at 0 (a large negative bias stops their gradient)"],
            ["<code>GELU</code>", "≈ [−0.17, ∞)", "Transformers; smooth version of ReLU", "Slightly more expensive"],
            ["<code>Tanh</code>", "(−1, 1)", "Small networks, recurrent cells, outputs in [−1, 1]", "Saturates: small gradients for large inputs"],
            ["<code>Sigmoid</code>", "(0, 1)", "The single output of a yes/no classifier, gates", "Saturates; not for hidden layers of deep nets"],
            ["<code>Softmax</code>", "(0, 1), rows sum to 1", "Turning class scores into probabilities <b>at inference</b>", "Do not train through it with CrossEntropy (" + ch("losses") + ")"],
        ], caption="Table 7.2 — Activation layers (all in NeuralSharp.Layers)"),
        honestbox("Why activations are needed, and where their math is",
                  "<p>A stack of <code>Linear</code> layers without activations collapses into a single linear map, "
                  "which cannot even learn XOR (" + ch("start") + "). The shapes, derivatives and gradient behaviour "
                  "of every activation (saturation, dead units, smoothness) are the subject of the Activation "
                  "Functions volume; the Glossary entries give one-line hints.</p>"),
        mex("the activations on a real task",
            "The same 1 → 32 → 32 → 1 network fitted to y = sin(3x) on [−1, 1] for 1,000 Adam steps, changing only "
            "the activation. Everything else, including the initial weights, is identical.",
            """
            float[] xs = [.. Enumerable.Range(0, 256).Select(i => i / 128f - 1f)];
            using var x = Tensor.From(xs, [256, 1]);
            using var y = Tensor.From([.. xs.Select(v => MathF.Sin(3 * v))], [256, 1]);

            foreach (var (name, make) in new (string, Func<Module>)[] {
                ("Sigmoid", () => new Sigmoid()), ("Tanh", () => new Tanh()),
                ("ReLU", () => new ReLU()), ("GELU", () => new GELU()) })
            {
                var r = new Random(3);
                using var model = new Sequential
                {
                    new Linear(1, 32, random: r), make(),
                    new Linear(32, 32, random: r), make(),
                    new Linear(32, 1, random: r),
                };
                using var opt = new Adam(model.Parameters(), learningRate: 0.01f);
                float loss = 0;
                for (int step = 1; step <= 1000; step++)
                {
                    using var scope = new TensorScope();
                    var l = Losses.MeanSquaredError(model.Forward(x), y);
                    opt.ZeroGrad(); l.Backward(); opt.Step();
                    if (step == 1000) loss = l.Item();
                }
                Console.WriteLine($"{name,-8} final loss {loss:F6}");
            }
            """,
            out="""
            Sigmoid  final loss 0.000126
            Tanh     final loss 0.000051
            ReLU     final loss 0.000005
            GELU     final loss 0.000035
            """,
            after="All four fit this smooth curve; ReLU gets closest in the step budget. On deeper networks the "
                  "gap between saturating (Sigmoid, Tanh) and non-saturating (ReLU, GELU) activations grows."),
        h2("7.3 The last layer decides what the network predicts"),
        reftable(["Task", "Last layer", "Loss (" + ch("losses") + ")", "Read predictions as"], [
            ["Regression, one or more numbers", "<code>Linear(h, k)</code>", "<code>MeanSquaredError</code> / <code>MeanAbsoluteError</code>", "the values (unscale them, " + ch("data") + ")"],
            ["Yes/no", "<code>Linear(h, 1)</code>", "<code>BinaryCrossEntropyWithLogits</code>", "<code>Sigmoid</code> of the output ≥ 0.5"],
            ["Yes/no (probabilities out)", "<code>Linear(h, 1)</code>, <code>Sigmoid</code>", "<code>BinaryCrossEntropy</code>", "the output ≥ 0.5"],
            ["One of K classes", "<code>Linear(h, K)</code>", "<code>SparseCrossEntropy</code> / <code>CrossEntropy</code>", "<code>ArgMax</code>; <code>Softmax</code> for probabilities"],
            ["Several independent yes/no labels", "<code>Linear(h, K)</code>", "<code>BinaryCrossEntropyWithLogits</code>", "<code>Sigmoid</code> of each output ≥ 0.5"],
            ["Values in a fixed range [a, b]", "<code>Linear(h, 1)</code>, <code>Sigmoid</code>", "MSE on targets scaled to [0, 1]", "a + (b − a)·output"],
        ], caption="Table 7.3 — Output layer, loss and interpretation"),
        trap("an activation after the last layer of a regression model",
             "<p><code>ReLU</code> after the output can never predict a negative value, and <code>Tanh</code> can never "
             "exceed 1. A regression network ends with a plain <code>Linear</code>.</p>"),
        h2("7.4 Designing an MLP"),
        deriv("A recipe that works for most tabular problems", [
            "Scale the inputs (" + ch("data") + "): standardized features train far better than raw ones.",
            "Start with two hidden layers: <code>Linear(F, 64)</code>, <code>ReLU</code>, <code>Linear(64, 64)</code>, <code>ReLU</code>, then the output layer from Table 7.3.",
            "Train with <code>Adam</code> at 10<sup>−3</sup> (" + ch("optimizers") + "). If the training loss stalls high, widen (128, 256) before adding layers.",
            "If the validation loss rises while the training loss falls (overfitting), add <code>Dropout(0.1–0.3)</code> after hidden activations or weight decay (" + ch("norm") + ", " + ch("optimizers") + ").",
            "Past three or four hidden layers, add <code>BatchNorm</code> or <code>LayerNorm</code>, or residual blocks (" + ch("modules") + ").",
        ]),
        cpugpu("an MLP for 20 features and 3 classes",
               """
               Device.Default = Device.Cpu;
               var r = new Random(42);
               using var model = new Sequential
               {
                   new Linear(20, 128, random: r), new ReLU(), new Dropout(0.2f, r),
                   new Linear(128, 128, random: r), new ReLU(), new Dropout(0.2f, r),
                   new Linear(128, 3, random: r),
               };
               """,
               """
               Device.Default = Device.Cuda();       // or: model.To(Device.Cuda()) after building
               // identical model code
               """,
               "At this size (about 19,000 parameters) the CPU is usually as fast as the GPU for batches under a few "
               "hundred rows; see " + ch("performance") + "."),
        reftable(["Layers", "Parameters", "Typical use"], [
            ["F → 32 → 1", "32F + 65", "Tiny problems, a baseline"],
            ["F → 64 → 64 → K", "64F + 4,160 + 65K + 64", "Most tabular tasks"],
            ["F → 256 → 256 → K", "256F + 65,792 + 257K + 256", "Large tabular data (100k+ rows)"],
        ], caption="Table 7.4 — Size of common MLPs (F features, K outputs)"),
        practice([
            (1, "Write the model for predicting 2 numbers from 12 features, with two hidden layers of 64 units.",
             "<code>new Sequential { new Linear(12, 64), new ReLU(), new Linear(64, 64), new ReLU(), new Linear(64, 2) }</code> "
             "trained with <code>MeanSquaredError</code>."),
            (1, "Why does a classifier's model end with <code>Linear(h, K)</code> and not with <code>Softmax</code> during training?",
             "The cross-entropy losses apply a numerically stable log-softmax themselves; adding <code>Softmax</code> first "
             "would apply it twice and slow learning. Use <code>Softmax</code> only to read probabilities at inference."),
            (2, "Repeat the sin(3x) experiment with a single hidden layer of 8 units. Which activation suffers most?",
             "Sigmoid suffers most. In our run (1,000 steps, seed 3) the final losses were Sigmoid 0.0118, ReLU 0.0019, "
             "Tanh 0.0002 and GELU 0.00003: with only 8 units, ReLU's piecewise-linear output (at most 8 kinks) fits a curve "
             "less closely than the smooth activations, and Sigmoid's non-zero-centred output slows learning."),
            (2, "Make two runs of the same training produce exactly the same numbers.",
             "Pass the same seeded <code>Random</code> to every layer (and to <code>Dropout</code>), shuffle data with a "
             "seeded <code>Random</code>, and keep the device the same. The CPU backend is deterministic; GPU results "
             "can differ in the last digits between runs because of parallel summation order."),
            (3, "Build a function <code>Sequential Mlp(int inputs, int[] hidden, int outputs, float dropout)</code> and use it to compare 1, 2 and 3 hidden layers on a problem of your choice.",
             "Create a <code>Sequential</code>, loop over <code>hidden</code> adding <code>Linear(prev, h)</code>, <code>ReLU</code> "
             "and (if dropout &gt; 0) <code>Dropout</code>, then add <code>Linear(prev, outputs)</code>. Compare validation "
             "losses rather than training losses."),
        ], PART),
        footer("Dense layer", "Linear", "Weight", "Bias", "Activation function", "ReLU", "GELU", "Tanh",
               "Sigmoid", "Softmax", "Saturation", "Initialization", "MLP", "Logits"),
    )
