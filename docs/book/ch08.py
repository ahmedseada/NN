"""Chapter 8 — Normalization and Dropout."""
from gen import *

PART = "II"


def norm_axes_svg():
    w, h = 470, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    def grid(ox, title, highlight):
        rows, cols = 4, 5
        for r in range(rows):
            for c in range(cols):
                on = highlight(r, c)
                p.append(f'<rect x="{ox + c * 30}" y="{30 + r * 24}" width="28" height="22" rx="2" '
                         f'fill="{"#0f6b5c" if on else "#e6f2ef"}" stroke="#0f6b5c" stroke-width="0.8"/>')
        p.append(svg_text(ox + 74, 18, title, 9, "#0f6b5c"))
        p.append(svg_text(ox + 74, 140, "rows = samples, columns = features", 7.4, "#56606a"))

    grid(20, "BatchNorm: one feature, all samples", lambda r, c: c == 1)
    grid(280, "LayerNorm: one sample, all features", lambda r, c: r == 2)
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "norm",
            "Normalization layers rescale activations inside the network so each layer sees inputs of a steady size; "
            "this lets deeper networks train faster and more reliably. Dropout randomly silences units during "
            "training so the network cannot rely on any single one, which reduces overfitting. All three layers "
            "behave differently in training and evaluation mode, which is the main thing to get right.",
            "<code>BatchNorm(channels)</code> normalizes each feature/channel over the batch; the usual choice for CNNs and MLPs with batches of 16+.",
            "<code>LayerNorm(features)</code> normalizes each sample over its features; the choice for transformers and small or variable batches.",
            "<code>Dropout(p)</code> zeroes a fraction p of values in training and does nothing in evaluation.",
            "BatchNorm keeps running averages (buffers) for evaluation; they are saved with the model.",
            "Always switch to <code>Eval()</code> for validation and inference (<code>Predict</code> and the Trainer do it for you).",
        ),
        h2("8.1 BatchNorm and LayerNorm: which values are normalized together"),
        diagram("Figure 8.1 — The two normalization directions", norm_axes_svg(),
                "The shaded cells are normalized together: brought to mean 0 and variance 1, then scaled and shifted by learned γ and β."),
        para("Both layers subtract a mean, divide by a standard deviation (glossary <b>Standardization</b>) and then "
             "apply a learned per-feature scale γ (<code>Gamma</code>, initially 1) and shift β (<code>Beta</code>, "
             "initially 0). They differ in which values share a mean: BatchNorm uses the same feature across the "
             "batch, LayerNorm uses all features of one sample."),
        reftable(["", "BatchNorm", "LayerNorm"], [
            ["Constructor", "<code>BatchNorm(channels, momentum = 0.1, epsilon = 1e-5)</code>", "<code>LayerNorm(features, epsilon = 1e-5)</code>"],
            ["Input", "<code>[N, C]</code> or <code>[N, C, ...]</code> (dimension 1 is normalized per channel)", "<code>[..., features]</code> (the last dimension)"],
            ["Statistics from", "the current batch (training) / running averages (evaluation)", "each sample, always"],
            ["Depends on batch size", "yes: unreliable below ~8 samples", "no"],
            ["Buffers", "<code>RunningMean</code>, <code>RunningVariance</code>", "none"],
            ["Typical place", "after Conv2d or Linear, before the activation", "before attention and feed-forward blocks (pre-norm), and before the output head"],
            ["Inference speed-up", "—", "one fused kernel when autograd is off"],
        ], caption="Table 8.1 — BatchNorm and LayerNorm"),
        mex("BatchNorm in training and in evaluation mode",
            "Two features on very different scales. In training mode each column is standardized with the batch's "
            "own statistics; the running averages move 10% (the momentum) towards them.",
            """
            var batch = Tensor.From(new float[,] { { 1, 100 }, { 2, 200 }, { 3, 300 }, { 4, 400 } });
            using var bn = new BatchNorm(2);

            Console.WriteLine(bn.Forward(batch));                 // training mode
            Console.WriteLine($"running mean {string.Join(", ", bn.RunningMean.ToArray())}");
            Console.WriteLine($"running var  {string.Join(", ", bn.RunningVariance.ToArray().Select(v => v.ToString("F2")))}");

            bn.Eval();
            Console.WriteLine(bn.Forward(batch));                 // evaluation mode
            """,
            out="""
            Tensor(shape=[4, 2], device=cpu, requiresGrad)
            [[-1.3416, -1.3416]
             [-0.4472, -0.4472]
             [0.4472, 0.4472]
             [1.3416, 1.3416]]
            running mean 0.25, 25
            running var  1.07, 1667.57
            Tensor(shape=[4, 2], device=cpu, requiresGrad)
            [[0.7262, 1.8366]
             [1.6944, 4.2855]
             [2.6627, 6.7343]
             [3.6309, 9.1831]]
            """,
            after="In training mode both columns come out identical: the scale difference is gone. In evaluation mode "
                  "the layer uses the running averages, which after a single batch are still far from the data (mean "
                  "0.25 instead of 2.5), so the output is not yet standardized. After a few hundred training steps the "
                  "running averages settle and the two modes agree closely. (The <code>requiresGrad</code> in the header "
                  "is because γ and β are parameters.)"),
        mex("LayerNorm normalizes each row on its own", None,
            """
            using var ln = new LayerNorm(4);
            var rows = Tensor.From(new float[,] { { 1, 2, 3, 4 }, { 10, 20, 30, 40 } });
            Console.WriteLine(ln.Forward(rows));
            """,
            out="""
            Tensor(shape=[2, 4], device=cpu, requiresGrad)
            [[-1.3416, -0.4472, 0.4472, 1.3416]
             [-1.3416, -0.4472, 0.4472, 1.3416]]
            """,
            after="The second row is ten times the first, yet both normalize to the same values: LayerNorm removes "
                  "each sample's own offset and scale. It needs no running statistics, so training and evaluation "
                  "behave the same."),
        trap("evaluating a BatchNorm model in training mode",
             "<p>In training mode BatchNorm uses the statistics of whatever batch it is given. Predicting a single "
             "sample then normalizes it against itself (every feature becomes 0), and predictions depend on which other "
             "samples share the batch. Call <code>Eval()</code>, or use <code>Predict</code>, for anything that is not "
             "a training step.</p>"),
        trap("tiny batches with BatchNorm",
             "<p>With batches of 1–4 the batch statistics are noisy and training suffers. Use larger batches, or "
             "switch to <code>LayerNorm</code>, which does not depend on the batch.</p>"),
        h2("8.2 Dropout"),
        para("During training, <code>Dropout(p)</code> sets each value to 0 with probability p and multiplies the "
             "survivors by 1/(1 − p), so the expected value is unchanged (glossary <b>Inverted dropout</b>). In "
             "evaluation mode it passes values through untouched. The mask is computed from a seed rather than "
             "stored, so dropout costs no extra memory and the backward pass regenerates the same mask."),
        mex("the same layer in both modes", None,
            """
            using var drop = new Dropout(0.5f, new Random(1));
            var ones = Tensor.Ones([2, 8]);
            Console.WriteLine(drop.Forward(ones));     // training: about half zeroed, the rest doubled
            drop.Eval();
            Console.WriteLine(drop.Forward(ones));     // evaluation: unchanged
            """,
            out="""
            Tensor(shape=[2, 8], device=cpu)
            [[0.0000, 0.0000, 0.0000, 2.0000, 2.0000, 0.0000, 2.0000, 2.0000]
             [2.0000, 0.0000, 2.0000, 2.0000, 2.0000, 0.0000, 0.0000, 0.0000]]
            Tensor(shape=[2, 8], device=cpu)
            [[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
             [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]
            """),
        reftable(["Where", "Typical p"], [
            ["After hidden activations of an MLP", "0.1–0.5 (0.2 is a good start)"],
            ["Before the classifier head of a CNN", "0.2–0.5"],
            ["Inside transformer blocks (<code>dropout:</code> argument)", "0.0–0.1"],
            ["Small datasets that overfit quickly", "towards the upper end"],
            ["Very large datasets, or underfitting", "0 (leave it out)"],
        ], caption="Table 8.2 — Dropout rates"),
        cpugpu("reproducible dropout",
               """
               var seed = new Random(7);
               using var model = new Sequential
               {
                   new Linear(16, 64, random: seed), new ReLU(), new Dropout(0.2f, seed),
                   new Linear(64, 1, random: seed),
               };
               """,
               """
               Device.Default = Device.Cuda();
               // same code: the mask is a hash of (seed, position), computed identically
               // by the CPU and GPU kernels, so both devices drop the same units
               """),
        h2("8.3 Where to put them"),
        snippet(f"""
            // MLP with BatchNorm (batches of 32+)
            new Linear(F, 128), new BatchNorm(128), new ReLU(), new Dropout(0.2f),
            new Linear(128, 128), new BatchNorm(128), new ReLU(), new Dropout(0.2f),
            new Linear(128, K),

            // CNN block ({ch("conv")})
            new Conv2d(32, 64, kernelSize: 3, padding: 1), new BatchNorm(64), new ReLU(), new MaxPool2d(2),

            // Transformer: TransformerEncoderLayer already contains two pre-norm LayerNorms;
            // add one final LayerNorm before the output head ({ch("attention")})
            """,
            caption="Standard placements"),
        honestbox("The bias before BatchNorm",
                  "<p>BatchNorm subtracts the per-channel mean, which cancels any bias added just before it. Passing "
                  "<code>bias: false</code> to the preceding <code>Linear</code> or <code>Conv2d</code> saves a few "
                  "parameters; leaving it in is harmless.</p>"),
        practice([
            (1, "Which layer would you use to normalize the 64-dimensional token vectors of a transformer, and why?",
             "<code>LayerNorm(64)</code>: it normalizes each token's own vector, does not depend on the batch size, "
             "and behaves the same in training and inference."),
            (1, "A model with Dropout gives different predictions for the same input on every call. What is wrong?",
             "It is being run in training mode. Call <code>model.Eval()</code> once (or use <code>Predict</code>), which "
             "turns dropout off."),
            (2, "What does BatchNorm output in training mode for a batch of one sample, and why?",
             "All zeros before γ/β (so β after them): each feature's batch mean is the sample itself and the deviation is 0. "
             "This is why single-sample inference must use evaluation mode."),
            (2, "Add BatchNorm to the MLP of " + ch("dense") + " (Section 7.4) and compare the training loss after 200 steps with and without it.",
             "Insert <code>new BatchNorm(128)</code> after each hidden <code>Linear</code>. With standardized inputs the "
             "difference on a shallow MLP is modest; it grows with depth and with poorly scaled inputs."),
            (3, "Estimate prediction uncertainty with Monte-Carlo dropout: run 50 predictions of the same input with dropout on and report their mean and spread.",
             "Keep the model in training mode (<code>model.Train()</code>) but wrap the calls in <code>Autograd.NoGrad()</code> "
             "and a <code>TensorScope</code>; call <code>Forward</code> 50 times, collect the outputs as floats, and "
             "compute their mean and standard deviation on the host. Do not use <code>Predict</code>, which switches "
             "dropout off. Note that BatchNorm layers, if any, would then also use batch statistics."),
        ], PART),
        footer("BatchNorm", "LayerNorm", "Standardization", "Running statistics", "Momentum (BatchNorm)",
               "Dropout", "Inverted dropout", "Overfitting", "Regularization", "Pre-norm"),
    )
