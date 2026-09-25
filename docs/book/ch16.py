"""Chapter 16 — Learning-Rate Schedules and Gradient Clipping."""
from gen import *

PART = "III"

RATES = {
    "StepDecay(3, 0.5)": [10.00, 10.00, 10.00, 5.00, 5.00, 5.00, 2.50, 2.50, 2.50, 1.25, 1.25],
    "ExponentialDecay(0.8)": [10.00, 8.00, 6.40, 5.12, 4.10, 3.28, 2.62, 2.10, 1.68, 1.34, 1.07],
    "CosineAnnealing(10)": [10.00, 9.76, 9.05, 7.94, 6.55, 5.00, 3.45, 2.06, 0.95, 0.24, 0.00],
    "Cosine, warm-up 2": [3.33, 6.67, 10.00, 9.62, 8.55, 6.94, 5.05, 3.16, 1.55, 0.48, 0.10],
}


def schedules_svg():
    w, h = 470, 170
    colors = ["#0f6b5c", "#a15c00", "#1f5fa8", "#b3261e"]
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    x0, y0, sx, sy = 40, 140, 26, 12
    p.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + 10 * sx}" y2="{y0}" stroke="#56606a"/>')
    p.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - 10 * sy}" stroke="#56606a"/>')
    for e in range(0, 11, 2):
        p.append(svg_text(x0 + e * sx, y0 + 12, str(e), 7.4, "#56606a"))
    p.append(svg_text(x0 + 5 * sx, y0 + 25, "epoch", 7.6, "#56606a"))
    p.append(svg_text(x0 - 6, y0 - 10 * sy + 4, "10", 7.4, "#56606a", "end"))
    p.append(svg_text(x0 - 6, y0 + 3, "0", 7.4, "#56606a", "end"))
    for i, (name, rates) in enumerate(RATES.items()):
        pts = " ".join(f"{x0 + e * sx},{y0 - r * sy:.1f}" for e, r in enumerate(rates))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i]}" stroke-width="1.6"/>')
        p.append(f'<rect x="{330}" y="{22 + i * 16}" width="10" height="3" fill="{colors[i]}"/>')
        p.append(svg_text(346, 26 + i * 16, name, 7.6, colors[i], "start"))
    p.append(svg_text(x0 + 2, 16, "learning rate ×10⁻³", 7.6, "#56606a", "start"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "schedules",
            "A single learning rate is rarely best for a whole run: large steps make fast early progress, small steps "
            "settle into a good minimum at the end. Schedules change the rate from epoch to epoch. Gradient clipping "
            "protects against the occasional huge gradient that would throw the model far off course. Both are one "
            "line with the Trainer.",
            "Schedules: <code>StepDecay</code>, <code>ExponentialDecay</code>, <code>CosineAnnealing</code> (with optional warm-up) and <code>LambdaSchedule</code> for anything else.",
            "A schedule wraps an optimizer; its current rate becomes the base rate. Call <code>Step()</code> once per epoch, or pass it to the Trainer.",
            "<code>CosineAnnealing(totalEpochs, warmupEpochs: …)</code> is a strong default, especially for transformers.",
            "<code>ClipGradientNorm(max)</code> rescales all gradients so their combined size is at most <code>max</code>; the Trainer does it with <code>MaxGradientNorm</code>.",
            "Clip for recurrent networks and transformers (max 1.0 is common).",
        ),
        h2("16.1 The schedules"),
        reftable(["Schedule", "Constructor", "Rate at epoch e"], [
            ["<code>StepDecay</code>", "<code>StepDecay(optimizer, stepSize, gamma = 0.1)</code>", "base · gamma<sup>" + brk("⌊") + "e / stepSize" + brk("⌋") + "</sup>"],
            ["<code>ExponentialDecay</code>", "<code>ExponentialDecay(optimizer, gamma)</code>", "base · gamma<sup>e</sup>"],
            ["<code>CosineAnnealing</code>", "<code>CosineAnnealing(optimizer, totalEpochs, minLearningRate = 0, warmupEpochs = 0)</code>", "linear warm-up, then a half cosine from base down to the minimum"],
            ["<code>LambdaSchedule</code>", "<code>LambdaSchedule(optimizer, (epoch, baseRate) =&gt; rate)</code>", "whatever your function returns"],
        ], caption="Table 16.1 — Learning-rate schedules (NeuralSharp.Optimizers)"),
        mex("printing each schedule for 11 epochs",
            "Each schedule wraps an <code>Adam</code> with rate 0.01; the numbers are the rate at the start of each "
            "epoch, in units of 10<sup>−3</sup>.",
            """
            using var w = Tensor.Zeros([1], requiresGrad: true);
            foreach (var (name, make) in new (string, Func<Optimizer, LearningRateScheduler>)[] {
                ("StepDecay(3, 0.5)",              o => new StepDecay(o, stepSize: 3, gamma: 0.5f)),
                ("ExponentialDecay(0.8)",          o => new ExponentialDecay(o, gamma: 0.8f)),
                ("CosineAnnealing(10)",            o => new CosineAnnealing(o, totalEpochs: 10)),
                ("Cosine(10, warmup 2, min 1e-4)", o => new CosineAnnealing(o, totalEpochs: 10, minLearningRate: 1e-4f, warmupEpochs: 2)),
                ("Lambda(1/(1+e))",                o => new LambdaSchedule(o, (e, lr) => lr / (1 + e))),
            })
            {
                using var opt = new Adam([w], learningRate: 0.01f);
                var schedule = make(opt);
                var rates = new List<string>();
                for (int e = 0; e < 11; e++)
                {
                    rates.Add((opt.LearningRate * 1000).ToString("F2"));
                    schedule.Step();
                }
                Console.WriteLine($"{name,-32} {string.Join(" ", rates)}");
            }
            """,
            out="""
            StepDecay(3, 0.5)                10.00 10.00 10.00 5.00 5.00 5.00 2.50 2.50 2.50 1.25 1.25
            ExponentialDecay(0.8)            10.00 8.00 6.40 5.12 4.10 3.28 2.62 2.10 1.68 1.34 1.07
            CosineAnnealing(10)              10.00 9.76 9.05 7.94 6.55 5.00 3.45 2.06 0.95 0.24 0.00
            Cosine(10, warmup 2, min 1e-4)   3.33 6.67 10.00 9.62 8.55 6.94 5.05 3.16 1.55 0.48 0.10
            Lambda(1/(1+e))                  10.00 5.00 3.33 2.50 2.00 1.67 1.43 1.25 1.11 1.00 0.91
            """),
        diagram("Figure 16.1 — Four schedules", schedules_svg(),
                "Warm-up starts small and climbs to the base rate before the cosine decay begins."),
        defbox("Warm-up",
               "<p>At the start of training, Adam's moment estimates are based on very few steps and the model's "
               "outputs are far from sensible, so full-size steps can be erratic. A short warm-up (a few percent of the "
               "run) ramps the rate up linearly first. It is standard for transformers and harmless elsewhere.</p>"),
        cpugpu("a schedule in a hand-written loop and with the Trainer",
               """
               using var optimizer = new AdamW(model.Parameters(), 1e-3f);
               var schedule = new CosineAnnealing(optimizer, totalEpochs: 50, warmupEpochs: 2);
               for (int epoch = 0; epoch < 50; epoch++)
               {
                   foreach (var batch in loader) { /* forward, loss, ZeroGrad, Backward, Step */ }
                   schedule.Step();                                   // once per epoch
               }
               """,
               """
               var trainer = new Trainer(model, optimizer, Losses.SparseCrossEntropy)
               {
                   Scheduler = new CosineAnnealing(optimizer, totalEpochs: 50, warmupEpochs: 2),
               };
               trainer.Fit(loader, epochs: 50);                       // steps the schedule for you
               """,
               "Schedules only set <code>optimizer.LearningRate</code>, a host-side number passed to the update kernel, "
               "so they behave identically on both devices."),
        trap("totalEpochs different from the epochs you train",
             "<p>If the cosine schedule is told 50 epochs but training runs 200, the rate reaches its minimum at epoch 50 "
             "and stays there for the remaining 150. Use the same number for both (early stopping may still end sooner).</p>"),
        h2("16.2 Gradient clipping"),
        para("Occasionally one batch produces a gradient hundreds of times larger than usual: a rare input, a long "
             "sequence in a recurrent network, a burst of large activations. One such step can undo hours of training. "
             "Clipping by the global norm measures the size of all gradients together (glossary <b>Gradient norm</b>) "
             "and, if it exceeds a limit, scales them all down by the same factor: the direction is kept, only the "
             "step length is limited."),
        mex("an extreme gradient, before and after clipping", None,
            """
            var r = new Random(1);
            using var model = new Sequential { new Linear(4, 8, random: r), new Tanh(), new Linear(8, 1, random: r) };
            using var sgd = new Sgd(model.Parameters(), 0.1f);
            using var scope = new TensorScope();

            var x = Tensor.Full([16, 4], 50f);                                // badly scaled input
            var loss = Losses.MeanSquaredError(model.Forward(x), Tensor.Full([16, 1], 1000f));
            sgd.ZeroGrad();
            loss.Backward();
            Console.WriteLine($"norm before {sgd.GradientNorm():F1}");
            double before = sgd.ClipGradientNorm(1f);
            Console.WriteLine($"ClipGradientNorm returned {before:F1}; norm now {sgd.GradientNorm():F4}");
            """,
            out="""
            norm before 132649.7
            ClipGradientNorm returned 132649.7; norm now 1.0000
            """),
        reftable(["How", "Code"], [
            ["By hand, between Backward and Step", "<code>loss.Backward(); optimizer.ClipGradientNorm(1f); optimizer.Step();</code>"],
            ["With the Trainer", "<code>new Trainer(…) { MaxGradientNorm = 1f }</code>"],
            ["Only measure (for logging)", "<code>double norm = optimizer.GradientNorm();</code>, or telemetry level <code>Gradients</code> (" + ch("telemetry") + ")"],
        ], caption="Table 16.2 — Clipping and measuring gradients"),
        honestbox("Clipping is not a cure for bad scaling",
                  "<p>The example above clips a gradient that exists only because the inputs were not scaled. Clipping "
                  "keeps such a run alive, but the real fix is to scale the data (" + ch("data") + "). Use clipping as "
                  "a safety net, typically with a limit of 0.5–5, and watch the gradient norm: if it is clipped on most "
                  "steps, the limit is too low or the learning rate too high.</p>"),
        practice([
            (1, "Write the schedule that halves the rate every 10 epochs.",
             "<code>new StepDecay(optimizer, stepSize: 10, gamma: 0.5f)</code>."),
            (1, "What rate does <code>CosineAnnealing(opt, totalEpochs: 10)</code> give at epoch 5 for a base rate of 0.01?",
             "Half the base rate, 0.005 (5.00 × 10⁻³ in the table): the cosine is halfway down."),
            (2, "Implement a one-cycle-style schedule with <code>LambdaSchedule</code>: rise linearly from 10% to 100% over the first 30% of 100 epochs, then fall linearly to 1%.",
             "<code>new LambdaSchedule(opt, (e, lr) =&gt; e &lt; 30 ? lr * (0.1f + 0.9f * e / 30f) : lr * (1f - 0.99f * (e - 30) / 70f))</code>."),
            (2, "Why is the norm reported by <code>ClipGradientNorm</code> the value before clipping?",
             "So you can log how large the gradients really were; after clipping the norm is simply the limit."),
            (3, "Train the LSTM memory task of " + ch("recurrent") + " with and without <code>MaxGradientNorm = 1f</code> at a learning rate of 0.05. Compare the loss curves.",
             "Use the Trainer with a <code>MetricsRecorder</code> (" + ch("telemetry") + ") and plot or print the loss per "
             "epoch. At a high rate, unclipped recurrent training tends to show sudden loss spikes; with clipping the curve "
             "is smoother."),
        ], PART),
        footer("Learning-rate schedule", "Step decay", "Cosine annealing", "Warm-up", "Gradient clipping",
               "Gradient norm", "Exploding gradient"),
    )
