"""Chapter 15 — Optimizers."""
from gen import *

PART = "III"


def build():
    return page(
        chapter_open(
            "optimizers",
            "After <code>Backward()</code>, every parameter holds a gradient. The optimizer decides how far to move "
            "each parameter in response. NeuralSharp has three: <code>Sgd</code> (with optional momentum), "
            "<code>Adam</code> and <code>AdamW</code>. All run their update as one fused kernel per parameter on "
            "either device. This chapter explains what each does, how to set the learning rate, weight decay, and "
            "how to train only part of a model.",
            "The step is always <code>ZeroGrad()</code>, <code>loss.Backward()</code>, <code>Step()</code>.",
            "<code>Adam(lr: 1e-3)</code> is the safe default; <code>AdamW</code> with weight decay for transformers and larger models.",
            "<code>Sgd</code> with momentum 0.9 can match Adam on CNNs with a tuned learning rate and schedule.",
            "The learning rate is the most important setting; <code>optimizer.LearningRate</code> can be changed at any time.",
            "Pass only the parameters you want trained: the optimizer never touches the others.",
        ),
        h2("15.1 The three optimizers"),
        reftable(["Optimizer", "Constructor (defaults)", "Update, in words"], [
            ["<code>Sgd</code>", "<code>Sgd(parameters, learningRate = 0.01, momentum = 0, weightDecay = 0)</code>",
             "Step against the gradient; with momentum, against a running sum of recent gradients"],
            ["<code>Adam</code>", "<code>Adam(parameters, learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0)</code>",
             "Divide a running mean of the gradient by the square root of a running mean of its square: every parameter gets its own step size"],
            ["<code>AdamW</code>", "<code>AdamW(parameters, learningRate = 0.001, …, weightDecay = 0.01)</code>",
             "Adam, plus shrinking every weight by lr · weightDecay each step, independently of the gradient"],
        ], caption="Table 15.1 — Optimizers (NeuralSharp.Optimizers)"),
        honestbox("The update formulas",
                  "<p>SGD: p ← p − lr · g (with momentum: v ← μv + g, p ← p − lr · v). Adam keeps two moving averages "
                  "(glossary <b>Exponential moving average</b>) and corrects them for their zero start; the idea is in "
                  "the glossary entry <b>Adam</b>. The derivations are outside this book; everything you need to use them "
                  "is the table above and the rules of thumb below.</p>"),
        reftable(["Member", "Purpose"], [
            ["<code>Parameters</code>", "The tensors this optimizer updates (fixed at construction)"],
            ["<code>LearningRate</code>", "Get or set the step size; schedules change it (" + ch("schedules") + ")"],
            ["<code>ZeroGrad()</code>", "Zero every parameter's gradient"],
            ["<code>Step()</code>", "Apply one update; parameters without a gradient are skipped"],
            ["<code>GradientNorm()</code>, <code>ClipGradientNorm(max)</code>", "Measure and limit the gradient size (" + ch("schedules") + ")"],
            ["<code>Dispose()</code>", "Free the optimizer's state (Adam's two moment tensors per parameter, SGD's velocity)"],
        ], caption="Table 15.2 — The Optimizer base class"),
        h2("15.2 Comparing them on one problem"),
        mex("SGD, momentum, Adam and AdamW fitting sin(3x)",
            "The Tanh network of " + ch("dense") + " (1 → 32 → 32 → 1), identical initial weights, 500 full-batch steps. "
            "The last column is the size (L2 norm) of all weights at the end.",
            """
            foreach (var (name, make) in new (string, Func<IEnumerable<Tensor>, Optimizer>)[] {
                ("Sgd(lr 0.1)",               p => new Sgd(p, learningRate: 0.1f)),
                ("Sgd(lr 0.1, momentum 0.9)", p => new Sgd(p, learningRate: 0.1f, momentum: 0.9f)),
                ("Adam(lr 0.01)",             p => new Adam(p, learningRate: 0.01f)),
                ("AdamW(lr 0.01, wd 0.1)",    p => new AdamW(p, learningRate: 0.01f, weightDecay: 0.1f)),
            })
            {
                var r = new Random(3);
                using var model = new Sequential
                {
                    new Linear(1, 32, random: r), new Tanh(),
                    new Linear(32, 32, random: r), new Tanh(),
                    new Linear(32, 1, random: r),
                };
                using var opt = make(model.Parameters());
                var line = $"{name,-27}";
                for (int step = 1; step <= 500; step++)
                {
                    using var scope = new TensorScope();
                    var loss = Losses.MeanSquaredError(model.Forward(x), y);
                    opt.ZeroGrad(); loss.Backward(); opt.Step();
                    if (step is 50 or 200 or 500) line += $" {step}: {loss.Item():F5}";
                }
                double norm = Math.Sqrt(model.Parameters().Sum(p => p.ToArray().Sum(v => (double)v * v)));
                Console.WriteLine($"{line}  |w| {norm:F2}");
            }
            """,
            out="""
            Sgd(lr 0.1)                 50: 0.14820 200: 0.04002 500: 0.01070  |w| 6.57
            Sgd(lr 0.1, momentum 0.9)   50: 0.04543 200: 0.00661 500: 0.00274  |w| 6.88
            Adam(lr 0.01)               50: 0.13338 200: 0.00461 500: 0.00096  |w| 7.94
            AdamW(lr 0.01, wd 0.1)      50: 0.13797 200: 0.00498 500: 0.00488  |w| 6.32
            """,
            after="Momentum makes SGD several times faster. Adam reaches the lowest loss. AdamW keeps the weights "
                  "smallest, at the cost of a slightly higher training loss: weight decay pulls towards simpler "
                  "solutions, which usually pays off on validation data rather than on the training data shown here."),
        h2("15.3 The learning rate"),
        reftable(["Symptom (training loss)", "Likely cause", "Try"], [
            ["Becomes NaN or explodes", "Rate far too high", "Divide by 10; clip gradients (" + ch("schedules") + ")"],
            ["Jumps around without falling", "Rate too high", "Divide by 3"],
            ["Falls very slowly and steadily", "Rate too low", "Multiply by 3"],
            ["Falls, then flattens early at a high value", "Rate fine, model too small or data unscaled", "Scale data, widen the model"],
            ["Falls well, then flattens", "Ready for a lower rate", "A decaying schedule (" + ch("schedules") + ")"],
        ], caption="Table 15.3 — Reading the loss curve"),
        reftable(["Optimizer", "Typical starting rate"], [
            ["Adam / AdamW, MLPs and CNNs", "1e-3 (3e-3 for small models)"],
            ["Adam / AdamW, transformers", "3e-4 to 1e-3, with warm-up"],
            ["SGD with momentum 0.9", "0.01 to 0.1"],
            ["Fine-tuning a trained model", "10× smaller than the original rate"],
        ], caption="Table 15.4 — Starting learning rates"),
        h2("15.4 Weight decay"),
        para("Weight decay shrinks weights slightly on every step, so the model only keeps large weights that clearly "
             "reduce the loss; it is a standard defence against overfitting (glossary <b>Weight decay</b>). "
             "<code>Sgd</code> and <code>Adam</code> apply it as an extra gradient term λ·p (classic L2); "
             "<code>AdamW</code> applies it directly to the weights, which works better with Adam's per-parameter "
             "step sizes. Typical values: 1e-4 to 1e-2 for AdamW."),
        cpugpu("AdamW for a transformer",
               """
               Device.Default = Device.Cpu;
               using var optimizer = new AdamW(model.Parameters(), learningRate: 1e-3f, weightDecay: 0.01f);
               """,
               """
               Device.Default = Device.Cuda();
               using var optimizer = new AdamW(model.Parameters(), learningRate: 1e-3f, weightDecay: 0.01f);
               // one fused adam kernel per parameter tensor per step; the moment tensors
               // are created on the parameter's device on the first Step()
               """),
        honestbox("What weight decay also shrinks",
                  "<p>The library applies weight decay to every parameter the optimizer holds, including biases and "
                  "normalization scales (γ). Many practitioners exempt those; with NeuralSharp you can do so by giving "
                  "them to a second optimizer without decay (Section 15.5).</p>"),
        h2("15.5 Training part of a model"),
        para("An optimizer updates exactly the tensors it was given. That makes freezing layers and using different "
             "rates for different parts straightforward; " + ch("finetune") + " builds on it."),
        snippet("""
            // Train only the last layer (e.g. a new head on a pretrained body):
            var head = (Linear)model[^1];
            using var optimizer = new Adam(head.Parameters(), learningRate: 1e-3f);

            // Faster rate for the head, slower for the body, no decay on 1-D tensors (biases, γ, β):
            var body = model.Parameters().Except(head.Parameters()).ToList();
            using var bodyDecay   = new AdamW(body.Where(p => p.Rank > 1), 1e-4f, weightDecay: 0.01f);
            using var bodyNoDecay = new Adam(body.Where(p => p.Rank == 1), 1e-4f);
            using var headOpt     = new AdamW(head.Parameters(), 1e-3f);
            Optimizer[] all = [bodyDecay, bodyNoDecay, headOpt];

            // in the loop:
            foreach (var o in all) o.ZeroGrad();
            loss.Backward();
            foreach (var o in all) o.Step();
            """.replace("model[^1]", "model[model.Count - 1]"), caption="Freezing and parameter groups"),
        trap("frozen parameters still get gradients",
             "<p>A parameter left out of the optimizer is not updated, but autograd still computes its gradient (it "
             "requires gradients). To save that work too, set <code>p.RequiresGrad = false</code> on the frozen "
             "parameters before training (" + ch("autograd") + ").</p>"),
        trap("recreating the optimizer every epoch",
             "<p>Adam's moving averages are its memory; a new optimizer starts from zero and takes large, noisy first "
             "steps. Create it once, next to the model, and dispose it at the end.</p>"),
        practice([
            (1, "Which optimizer and learning rate would you start with for a new tabular MLP?",
             "<code>Adam</code> at 1e-3 (or <code>AdamW</code> with weight decay 1e-4 if it overfits)."),
            (1, "Your loss becomes NaN after a few steps. List two fixes.",
             "Lower the learning rate (÷10) and clip the gradient norm; also check that inputs are scaled and targets "
             "have no NaN values."),
            (2, "Rerun Section 15.2 with <code>Adam(lr 0.1)</code> and <code>Adam(lr 0.001)</code>. What do you observe?",
             "Measured: at 0.1 the loss was 0.036, 0.0057 and 0.00014 at steps 50, 200 and 500, the lowest of all runs, "
             "with much larger weights (|w| 12.3); at 0.001 it was still 0.0074 after 500 steps. On a small, noise-free, "
             "full-batch problem a high rate can win; with noisy mini-batches it is much riskier, which is why 1e-3 is "
             "the usual starting point."),
            (2, "How many extra floats does Adam store for a model with 1,000,000 parameters, and how many bytes?",
             "Two moment tensors of the same size: 2,000,000 floats = 8 MB (plus 4 MB for the gradients)."),
            (3, "Write the training loop for a two-part model (encoder + head) where the encoder is frozen for the first 5 epochs and then trained at a 10× smaller rate.",
             "Create <code>headOpt</code> over the head's parameters from the start. For epochs 1–5 set "
             "<code>RequiresGrad = false</code> on the encoder's parameters and step only <code>headOpt</code>. At epoch 6 set "
             "<code>RequiresGrad = true</code> again, create <code>encoderOpt</code> with rate lr/10, and from then on zero and "
             "step both."),
        ], PART),
        footer("Optimizer", "SGD", "Momentum", "Adam", "AdamW", "Learning rate", "Weight decay",
               "Exponential moving average", "Parameter group", "Freezing"),
    )
