"""Chapter 5 — Autograd: How Gradients Are Computed."""
from gen import *

PART = "I"


def graph_svg():
    w, h = 480, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    def node(x, y, label, leaf=False):
        fill = "#e6f2ef" if leaf else "#ffffff"
        p.append(f'<rect x="{x - 32}" y="{y - 14}" width="64" height="28" rx="5" fill="{fill}" stroke="#0f6b5c" stroke-width="1.2"/>')
        p.append(svg_text(x, y + 4, label, 8.8, "#0f6b5c"))

    def fwd(x1, y1, x2, y2):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#56606a" stroke-width="1.1"/>')

    node(40, 40, "x", True)
    node(40, 100, "W", True)
    node(130, 70, "MatMul")
    node(130, 128, "b", True)
    node(220, 70, "+ b")
    node(310, 70, "Tanh")
    node(400, 70, "Mean")
    fwd(72, 40, 98, 64); fwd(72, 100, 98, 76); fwd(162, 70, 188, 70); fwd(162, 128, 196, 84)
    fwd(252, 70, 278, 70); fwd(342, 70, 368, 70)
    p.append('<line x1="432" y1="70" x2="462" y2="70" stroke="#56606a" stroke-width="1.1"/>')
    p.append(svg_text(463, 60, "loss", 8.4, "#1a1f23", "end"))
    p.append('<path d="M 400 44 C 330 8, 150 8, 130 44" fill="none" stroke="#b3261e" stroke-width="1.2" stroke-dasharray="4 3"/>')
    p.append('<polygon points="130,48 126,40 134,40" fill="#b3261e"/>')
    p.append(svg_text(265, 16, "Backward(): visit nodes in reverse, add each input's gradient", 7.6, "#b3261e"))
    p.append(svg_text(40, 140, "leaves (shaded)", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "autograd",
            "Training needs, for every parameter, the direction in which the loss grows. NeuralSharp computes these "
            "gradients automatically: while you compute the forward pass, it records which operation produced which "
            "tensor, and <code>Backward()</code> replays that record in reverse. This chapter shows exactly what is "
            "recorded, when, and how to control it: turning it off, cutting it, accumulating and checking gradients.",
            "Set <code>requiresGrad: true</code> on the tensors you want gradients for; layers do this for their parameters.",
            "<code>loss.Backward()</code> fills <code>.Grad</code> of every tensor that needs one. Gradients <b>add up</b> until you zero them.",
            "The recorded graph is freed by <code>Backward()</code>: to differentiate again, run the forward pass again.",
            "<code>Autograd.NoGrad()</code> stops recording (inference, evaluation); <code>Detach()</code> cuts one tensor off.",
            "Autograd works identically on both devices; gradients live on the device of their tensor.",
        ),
        h2("5.1 Leaves, nodes and the graph"),
        para("A <b>leaf</b> is a tensor you created directly: data, targets, or a parameter. A tensor produced by an "
             "operation is a <b>node</b>. When at least one input of an operation has <code>RequiresGrad</code> set "
             "and recording is on, the output remembers its inputs and a small function that knows how to pass a "
             "gradient back to them. The chain of these records is the <b>computation graph</b>."),
        diagram("Figure 5.1 — The graph of tanh(x·W + b) averaged into a loss", graph_svg(),
                "Forward runs left to right and records each step; Backward walks the same steps right to left."),
        reftable(["Property / method", "Meaning"], [
            ["<code>RequiresGrad</code>", "Whether gradients flow to this tensor. Settable only on leaves; nodes inherit it."],
            ["<code>IsLeaf</code>", "True when the tensor was not produced by a recorded operation."],
            ["<code>Grad</code>", "The accumulated gradient (same shape and device), or null before any Backward."],
            ["<code>Backward()</code>", "Starts from a one-element tensor (a loss) with gradient 1."],
            ["<code>Backward(gradient)</code>", "Starts from any tensor with a given gradient of the same shape."],
            ["<code>ZeroGrad()</code>", "Sets this tensor's gradient to zero (optimizers zero all their parameters at once)."],
            ["<code>Detach()</code>", "Same data, no history, <code>RequiresGrad</code> false."],
            ["<code>Autograd.NoGrad()</code>", "Disposable scope: no recording on this thread while it is open."],
            ["<code>Autograd.IsEnabled</code>", "Whether recording is currently on for this thread."],
        ], caption="Table 5.1 — The autograd API"),
        h2("5.2 A first gradient"),
        mex("the derivative of a sum of squares",
            "For y = x₁² + x₂² + x₃² the gradient is 2x (glossary <b>Gradient</b>). Autograd reproduces it exactly.",
            """
            using var x = Tensor.From([1f, 2f, 3f], requiresGrad: true);
            using (var scope = new TensorScope())
            {
                var y = (x * x).Sum();
                y.Backward();
            }
            Console.WriteLine(x.Grad);            // gradients are never captured by a scope
            """,
            out="""
            Tensor(shape=[3], device=cpu)
            [2.0000, 4.0000, 6.0000]
            """),
        deriv("What Backward() does", [
            "Checks that the start tensor has one element (or that you passed a gradient of its shape).",
            "Seeds that tensor's gradient with 1.",
            "Orders the recorded nodes so that every node comes before the nodes it was computed from.",
            "For each node, calls its recorded backward function, which <b>adds</b> into the gradients of its inputs "
            "(glossary <b>Chain rule</b> for why products of local derivatives appear here).",
            "Frees each intermediate node's gradient and record as soon as it has been used; leaves keep their <code>Grad</code>.",
        ]),
        h2("5.3 Gradients accumulate"),
        para("Backward adds into <code>Grad</code> instead of overwriting it. This is what lets a tensor used in "
             "several places receive the sum of all its contributions, and it is why every training step starts with "
             "<code>optimizer.ZeroGrad()</code>."),
        mex("accumulation and ZeroGrad", None,
            """
            using (var scope = new TensorScope()) { (x * x).Sum().Backward(); }   // a second pass
            Console.WriteLine(x.Grad);             // 2x + 2x
            x.ZeroGrad();
            Console.WriteLine(x.Grad);
            """,
            out="""
            Tensor(shape=[3], device=cpu)
            [4.0000, 8.0000, 12.0000]
            Tensor(shape=[3], device=cpu)
            [0.0000, 0.0000, 0.0000]
            """),
        snippet("""
            // Effective batch of 4 × 64 rows when only 64 fit in memory.
            const int parts = 4;
            optimizer.ZeroGrad();
            foreach (var (xPart, yPart) in Split(x, y, parts))      // your own splitting helper
            {
                using var scope = new TensorScope();
                var loss = Losses.MeanSquaredError(model.Forward(xPart), yPart) * (1f / parts);
                loss.Backward();                                      // gradients add up
            }
            optimizer.Step();                                         // one update for all parts
            """, caption="Using accumulation on purpose: gradient accumulation for large batches"),
        h2("5.4 Turning recording off, and cutting the graph"),
        mex("NoGrad and Detach", None,
            """
            using (var scope = new TensorScope())
            {
                var h = x * 2f;
                Console.WriteLine($"x leaf {x.IsLeaf}, h leaf {h.IsLeaf}, h requires grad {h.RequiresGrad}");
                using (Autograd.NoGrad())
                {
                    var p = x * 2f;
                    Console.WriteLine($"under NoGrad: requires grad {p.RequiresGrad}");
                }
                var d = h.Detach();
                Console.WriteLine($"detached: leaf {d.IsLeaf}, requires grad {d.RequiresGrad}");
            }
            """,
            out="""
            x leaf True, h leaf False, h requires grad True
            under NoGrad: requires grad False
            detached: leaf True, requires grad False
            """),
        reftable(["Tool", "Use it when", "Effect"], [
            ["<code>Autograd.NoGrad()</code>", "Inference, evaluation, metrics, anything you will not differentiate", "Nothing is recorded: faster, no graph memory"],
            ["<code>model.Predict(x)</code>", "Inference through a module", "NoGrad + evaluation mode + scope, in one call"],
            ["<code>t.Detach()</code>", "Using a value as a constant (e.g. a target computed by the model itself)", "Gradients stop at that tensor"],
            ["<code>p.RequiresGrad = false</code> on parameters", "Freezing layers when fine-tuning (" + ch("finetune") + ")", "No gradient is computed for them"],
        ], caption="Table 5.2 — Controlling what is recorded"),
        trap("calling Backward on something computed under NoGrad",
             "<p>It throws <i>This tensor does not require gradients: none of its inputs had RequiresGrad = true (or it "
             "was computed under Autograd.NoGrad())</i>. Compute the loss outside the NoGrad block.</p>"),
        trap("calling Backward twice on the same loss",
             "<p>The graph is freed by the first call. A second call does not throw, but it no longer reaches the "
             "parameters, so their gradients do not change. Run the forward pass again for another gradient.</p>"),
        h2("5.5 Worked example: fitting a line with raw tensors"),
        para("Layers and the Trainer are conveniences over what follows. Here the model is just two parameters, "
             "<code>w</code> and <code>b</code>, fitted to noisy points on the line y = 3x + 2. Any optimizer accepts "
             "any list of tensors that require gradients."),
        cpugpu("fitting y = 3x + 2",
               """
               Device.Default = Device.Cpu;
               var rng = new Random(0);
               float[] xs = [.. Enumerable.Range(0, 64).Select(i => i / 32f - 1f)];
               float[] ys = [.. xs.Select(v => 3f * v + 2f + 0.05f * (rng.NextSingle() - 0.5f))];

               using var inputs  = Tensor.From(xs, [64, 1]);
               using var targets = Tensor.From(ys, [64, 1]);
               using var w = Tensor.Zeros([1, 1], requiresGrad: true);
               using var b = Tensor.Zeros([1], requiresGrad: true);
               using var sgd = new Sgd([w, b], learningRate: 0.5f);

               for (int step = 1; step <= 60; step++)
               {
                   using var scope = new TensorScope();
                   var error = inputs.MatMul(w) + b - targets;
                   var loss = (error * error).Mean();
                   sgd.ZeroGrad();
                   loss.Backward();
                   sgd.Step();
                   if (step % 20 == 0)
                       Console.WriteLine($"step {step}: loss {loss.Item():F6}  w {w.Item():F4}  b {b.Item():F4}");
               }
               """,
               """
               Device.Default = Device.Cuda();      // the only change
               """,
               "Output (the same on both devices, up to the last printed digit):"),
        output("""
            step 20: loss 0.000204  w 2.9984  b 2.0036
            step 40: loss 0.000203  w 2.9993  b 2.0037
            step 60: loss 0.000203  w 2.9993  b 2.0037
            """),
        para("The fitted line is y = 2.9993x + 2.0037; the remaining loss is the noise that was added on purpose. "
             "Swapping <code>Sgd</code> for <code>Adam</code>, or the two tensors for a <code>Linear(1, 1)</code> layer, "
             "changes nothing else in the loop."),
        h2("5.6 Checking a gradient numerically"),
        para("When you write a custom layer or loss from tensor operations, you can check its gradients against "
             "finite differences: nudge one input up and down by a small h, and compare the change in output with "
             "what autograd says (glossary <b>Finite difference</b>). The library's own test suite checks every "
             "operation this way on both devices (" + ch("testing") + ")."),
        mex("autograd against finite differences for f(x) = Σ x·tanh(x)", None,
            """
            using var p0 = Tensor.From([0.3f, -1.2f, 2.0f], requiresGrad: true);
            static Tensor F(Tensor t) => (t.Tanh() * t).Sum();

            using (var scope = new TensorScope()) { F(p0).Backward(); }
            float[] analytic = p0.Grad!.ToArray();
            float[] values = p0.ToArray();
            const float eps = 1e-3f;

            for (int i = 0; i < values.Length; i++)
            {
                float Eval(float delta)
                {
                    var shifted = (float[])values.Clone();
                    shifted[i] += delta;
                    using var scope = new TensorScope();
                    using var noGrad = Autograd.NoGrad();
                    return F(Tensor.From(shifted)).Item();
                }
                float numeric = (Eval(eps) - Eval(-eps)) / (2 * eps);
                Console.WriteLine($"x{i}: autograd {analytic[i]:F4}  numeric {numeric:F4}");
            }
            """,
            out="""
            x0: autograd 0.5659  numeric 0.5659
            x1: autograd -1.1997  numeric -1.1997
            x2: autograd 1.1053  numeric 1.1054
            """),
        honestbox("Precision of the check",
                  "<p>Tensors are float32, so finite differences are only good to about three or four significant "
                  "digits; agreement to within about 1% is a pass. Use a step h near 10<sup>−3</sup>: much smaller and "
                  "rounding error dominates, much larger and the curvature of the function does.</p>"),
        h2("5.7 What is differentiable"),
        reftable(["Differentiable", "Not differentiable (no gradient flows)"], [
            ["All operators, element-wise functions, <code>Dropout</code>", "<code>ArgMax</code>, <code>OneHot</code>"],
            ["<code>Sum</code>, <code>Mean</code> (all or along a dimension), <code>Softmax</code>, <code>LogSoftmax</code>", "Values read to the host (<code>Item</code>, <code>ToArray</code>) and turned back into tensors"],
            ["<code>MatMul</code> (all shape patterns and transposes)", "The fused inference kernels used for fast generation (" + ch("generation") + "); they run only under NoGrad"],
            ["<code>Reshape</code>, <code>Flatten</code>, <code>Permute</code>, <code>Transpose</code>, <code>Narrow</code>, <code>Concat</code>, <code>Stack</code>", "Metrics"],
            ["Every layer's forward pass and every loss", ""],
        ], caption="Table 5.3 — Where gradients flow"),
        practice([
            (1, "Compute the gradient of <code>(x * x * x).Sum()</code> at x = [1, 2] and check it against 3x².",
             "Autograd gives [3, 12]. The product <code>x * x * x</code> uses x three times; the contributions add up."),
            (1, "Why does <code>x.Grad</code> survive the <code>TensorScope</code> in Section 5.2, while <code>y</code> does not?",
             "Gradients are allocated outside any scope (they belong to their tensor), while <code>y</code> was created "
             "inside the scope and is disposed with it."),
            (2, "Fit y = 3x + 2 with <code>Adam</code> at learning rate 0.1 instead of <code>Sgd</code>. How many steps does it need?",
             "Replace the optimizer line with <code>new Adam([w, b], learningRate: 0.1f)</code>. Adam's step size does not "
             "depend on the gradient's scale, so it approaches the solution steadily and needs more steps here than "
             "SGD with a well-chosen rate (typically 100–200)."),
            (2, "Use <code>Backward(gradient)</code> to get the gradient of each output of <code>x.Tanh()</code> with respect to x at once.",
             "Because each output depends only on its own input, <code>y = x.Tanh(); y.Backward(Tensor.Ones(y.Shape));</code> "
             "gives the element-wise derivative 1 − tanh²(x) in <code>x.Grad</code>."),
            (3, "Write a numeric gradient checker <code>bool Check(Func&lt;Tensor, Tensor&gt; f, float[] at)</code> that returns true when every component agrees within 1%.",
             "Generalize Section 5.6: compute the analytic gradient once, then for each index compare with the central "
             "difference; accept when |a − n| ≤ 0.01·max(1, |a|, |n|). Run it on both devices by creating the tensors "
             "with a <code>device</code> argument."),
        ], PART),
        footer("Autograd", "Leaf", "Computation graph", "Gradient", "Chain rule", "Backward pass", "NoGrad",
               "Detach", "Gradient accumulation", "Finite difference", "SGD"),
    )
