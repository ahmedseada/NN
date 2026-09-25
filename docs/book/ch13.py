"""Chapter 13 — Losses."""
from gen import *

PART = "III"


def build():
    return page(
        chapter_open(
            "losses",
            "The loss turns a batch of predictions and targets into one number that training drives down. Choosing "
            "it is choosing what \"good\" means for your model. NeuralSharp has six losses in the static class "
            "<code>Losses</code>; each returns a scalar tensor you call <code>Backward()</code> on. This chapter shows "
            "what each measures, which output layer and target format it expects, and how to write your own.",
            "Regression: <code>MeanSquaredError</code> (default) or <code>MeanAbsoluteError</code> (robust to outliers).",
            "Classes: <code>SparseCrossEntropy(logits, classIds)</code> or <code>CrossEntropy(logits, oneHot)</code>, both from raw scores.",
            "Yes/no: <code>BinaryCrossEntropyWithLogits(logits, targets)</code>; use <code>BinaryCrossEntropy</code> only after a Sigmoid.",
            "<code>labelSmoothing</code> (e.g. 0.1) makes classifiers less over-confident.",
            "A custom loss is any function <code>(Tensor predictions, Tensor targets) =&gt; Tensor</code> built from tensor operations.",
        ),
        h2("13.1 The six losses"),
        reftable(["Loss", "Predictions", "Targets", "Computes"], [
            ["<code>MeanSquaredError(p, t)</code>", "any shape", "same shape", "mean((p − t)²)"],
            ["<code>MeanAbsoluteError(p, t)</code>", "any shape", "same shape", "mean(|p − t|)"],
            ["<code>CrossEntropy(logits, t, labelSmoothing = 0)</code>", "<code>[..., K]</code> raw scores", "<code>[..., K]</code> one-hot or probabilities", "−mean over rows of Σ t · log softmax(logits)"],
            ["<code>SparseCrossEntropy(logits, ids, labelSmoothing = 0)</code>", "<code>[..., K]</code> raw scores", "<code>[...]</code> class ids", "the same, without building one-hot targets yourself"],
            ["<code>BinaryCrossEntropyWithLogits(z, t)</code>", "any shape, raw scores", "same shape, 0 or 1", "binary cross-entropy of sigmoid(z), computed stably"],
            ["<code>BinaryCrossEntropy(p, t)</code>", "probabilities in (0, 1)", "same shape, 0 or 1", "−mean(t·log p + (1 − t)·log(1 − p))"],
        ], caption="Table 13.1 — Losses (static class NeuralSharp.Losses)"),
        honestbox("Where the math is",
                  "<p>Squared and absolute errors need nothing beyond the Pre-Calc volume. Cross-entropy is the average "
                  "negative log-probability the model gave to the right answer (glossary <b>Cross-entropy</b>); the "
                  "softmax and sigmoid it builds on, and why their gradients pair so neatly with it, are in the "
                  "Activation Functions volume.</p>"),
        h2("13.2 Regression: squared or absolute error"),
        mex("the same errors under both losses, with and without an outlier", None,
            """
            var prediction = Tensor.From(new float[,] { { 2.5f }, { 0.0f }, { 2.0f }, { 8.0f } });
            var target     = Tensor.From(new float[,] { { 3.0f }, { -0.5f }, { 2.0f }, { 7.0f } });
            Console.WriteLine($"MSE {Losses.MeanSquaredError(prediction, target).Item():F4}  " +
                              $"MAE {Losses.MeanAbsoluteError(prediction, target).Item():F4}");

            var outlier = Tensor.From(new float[,] { { 2.5f }, { 0.0f }, { 2.0f }, { 17.0f } });
            Console.WriteLine($"with an outlier: MSE {Losses.MeanSquaredError(outlier, target).Item():F4}  " +
                              $"MAE {Losses.MeanAbsoluteError(outlier, target).Item():F4}");
            """,
            out="""
            MSE 0.3750  MAE 0.5000
            with an outlier: MSE 25.1250  MAE 2.7500
            """,
            after="One bad prediction multiplies MSE by 67 but MAE only by 5.5. MSE pushes hard to fix large errors; "
                  "MAE treats every unit of error the same, so a few extreme targets do not dominate training."),
        reftable(["Choose", "When"], [
            ["<code>MeanSquaredError</code>", "The default; errors are roughly normal; large errors are much worse than small ones"],
            ["<code>MeanAbsoluteError</code>", "Data has outliers or heavy tails (prices, durations); you report MAE anyway"],
            ["MSE on log-targets", "Targets span orders of magnitude (1 to 1,000,000): train on log(y), predict exp(output)"],
        ], caption="Table 13.2 — Regression losses"),
        h2("13.3 Classification: cross-entropy"),
        para("Both cross-entropy losses take <b>raw scores</b> (logits) straight from the last <code>Linear</code> "
             "and apply a numerically stable log-softmax internally. Do not put a <code>Softmax</code> layer before "
             "them. <code>SparseCrossEntropy</code> takes class ids; it works for any number of leading dimensions, "
             "so per-token predictions <code>[N, T, V]</code> with ids <code>[N, T]</code> need no reshaping."),
        mex("cross-entropy values that are worth remembering", None,
            """
            var logits  = Tensor.From(new float[,] { { 2.0f, 0.5f, -1.0f }, { 0.1f, 0.2f, 3.0f } });
            var classes = Tensor.From([0f, 2f]);                          // both rows predicted correctly
            var oneHot  = Tensor.From(new float[,] { { 1, 0, 0 }, { 0, 0, 1 } });

            Console.WriteLine($"sparse {Losses.SparseCrossEntropy(logits, classes).Item():F4}  " +
                              $"one-hot {Losses.CrossEntropy(logits, oneHot).Item():F4}  " +
                              $"smoothed(0.1) {Losses.SparseCrossEntropy(logits, classes, labelSmoothing: 0.1f).Item():F4}");
            Console.WriteLine($"wrong labels: {Losses.SparseCrossEntropy(logits, Tensor.From([2f, 0f])).Item():F4}");
            Console.WriteLine($"know-nothing (uniform) {Losses.SparseCrossEntropy(Tensor.Zeros([2, 3]), classes).Item():F4}  " +
                              $"ln 3 = {MathF.Log(3):F4}");
            Console.WriteLine($"per-token {Losses.SparseCrossEntropy(Tensor.Zeros([4, 10, 65]), Tensor.Zeros([4, 10])).Item():F4}  " +
                              $"ln 65 = {MathF.Log(65):F4}");
            """,
            out="""
            sparse 0.1755  one-hot 0.1755  smoothed(0.1) 0.3455
            wrong labels: 3.1255
            know-nothing (uniform) 1.0986  ln 3 = 1.0986
            per-token 4.1744  ln 65 = 4.1744
            """),
        defbox("The starting value of cross-entropy",
               "<p>A model that knows nothing gives every one of K classes the same probability, and its cross-entropy "
               "is ln K: 1.0986 for 3 classes, 2.3026 for 10, 4.1744 for a 65-character vocabulary. A freshly "
               "initialized classifier should start near ln K; the loss should then fall below it quickly. A loss far "
               "above ln K at the start usually means wrong labels or unscaled inputs.</p>"),
        para("<b>Label smoothing</b> replaces the one-hot target by (1 − s)·one-hot + s/K, so the model is rewarded "
             "for 90% confidence rather than 100% (with s = 0.1). The reported loss is higher, as above, but the "
             "model is usually better calibrated and generalizes slightly better. The " + ch("gpt") + " GPT trains "
             "without it; image classifiers often use 0.1."),
        h2("13.4 Yes/no: binary cross-entropy"),
        mex("why the logits version is preferred",
            "Four raw scores, two of them extreme and wrong (30 for a 0 target, −30 for a 1 target).",
            """
            var z = Tensor.From(new float[,] { { 3f }, { -2f }, { 30f }, { -30f } });
            var y = Tensor.From(new float[,] { { 1f }, { 0f }, { 0f }, { 1f } });
            Console.WriteLine($"with logits {Losses.BinaryCrossEntropyWithLogits(z, y).Item():F4}  " +
                              $"on probabilities {Losses.BinaryCrossEntropy(z.Sigmoid(), y).Item():F4}");

            var z2 = Tensor.From(new float[,] { { 3f }, { -2f } });
            var y2 = Tensor.From(new float[,] { { 1f }, { 0f } });
            Console.WriteLine($"moderate: with logits {Losses.BinaryCrossEntropyWithLogits(z2, y2).Item():F4}  " +
                              $"on probabilities {Losses.BinaryCrossEntropy(z2.Sigmoid(), y2).Item():F4}");
            """,
            out="""
            with logits 15.0439  on probabilities 8.1029
            moderate: with logits 0.0878  on probabilities 0.0878
            """,
            after="For moderate scores the two agree. For confident mistakes the sigmoid rounds to exactly 0 or 1 in "
                  "float32, the probability version has to clip (it adds 10<sup>−7</sup> inside the log) and it both "
                  "under-reports the loss and loses the gradient that would correct the mistake. The logits version "
                  "computes the same quantity without ever forming the rounded probability."),
        cpugpu("a binary classifier: model, loss and prediction",
               """
               Device.Default = Device.Cpu;
               using var model = new Sequential { new Linear(F, 32), new ReLU(), new Linear(32, 1) };  // no Sigmoid
               Func<Tensor, Tensor, Tensor> loss = Losses.BinaryCrossEntropyWithLogits;
               // at inference: probability = model.Predict(x).Sigmoid()
               """,
               """
               Device.Default = Device.Cuda();
               // identical; for accuracy on logits use Metric.BinaryAccuracy(threshold: 0f) (Chapter 14)
               """.replace("Chapter 14", ch("metrics"))),
        h2("13.5 Your own loss"),
        para("The <code>Trainer</code> accepts any <code>Func&lt;Tensor, Tensor, Tensor&gt;</code>. Build it from "
             "differentiable tensor operations (" + ch("autograd") + ", Table 5.3) and it trains like the built-in ones."),
        snippet("""
            // Huber loss: squared for small errors, absolute for large ones (delta = 1).
            // 0.5·e² when |e| ≤ 1, |e| − 0.5 otherwise, written without comparisons:
            // with c = min(|e|, 1):  loss = 0.5·c² + (|e| − c)
            static Tensor Huber(Tensor prediction, Tensor target)
            {
                var abs = (prediction - target).Abs();
                var clipped = 1f - (1f - abs).Relu();                 // min(|e|, 1)
                return (0.5f * clipped.Square() + (abs - clipped)).Mean();
            }

            // Weighted MSE: some outputs matter more (weights: [K], used as a [K, 1] column)
            static Func<Tensor, Tensor, Tensor> WeightedMse(Tensor weights) => (p, t) =>
                ((p - t).Square().MatMul(weights.Reshape(-1, 1))).Mean();

            var trainer = new Trainer(model, optimizer, Huber);
            """, caption="Two custom losses"),
        trap("Softmax before cross-entropy",
             "<p>A model ending in <code>Softmax</code> trained with <code>CrossEntropy</code> applies softmax twice. "
             "It still runs, but learning is slow and the loss plateaus high. Remove the layer; apply "
             "<code>Softmax()</code> only when you need probabilities after training.</p>"),
        trap("targets in the wrong format",
             "<p><code>CrossEntropy</code> needs one-hot rows of the logits' shape; <code>SparseCrossEntropy</code> needs "
             "ids with the logits' shape minus the last dimension. <code>Dataset.FromClassLabels</code> produces "
             "one-hot targets; a CSV with a class-id column can use either <code>ToOneHot(K)</code> + "
             "<code>CrossEntropy</code>, or the ids directly with <code>SparseCrossEntropy</code> (" + ch("data") + ").</p>"),
        practice([
            (1, "A 10-class classifier reports a loss of 2.30 after its first epoch. What does that tell you?",
             "2.30 ≈ ln 10: the model is still no better than guessing uniformly. Check labels, input scaling and the "
             "learning rate if it does not fall soon."),
            (1, "Which loss and output layer for predicting tomorrow's temperature?",
             "A plain <code>Linear(h, 1)</code> output with <code>MeanSquaredError</code> (or <code>MeanAbsoluteError</code> "
             "if the data has outliers), on standardized targets."),
            (2, "Show that <code>SparseCrossEntropy(logits, ids)</code> equals <code>CrossEntropy(logits, Tensor.OneHot(ids, K))</code> on a random batch.",
             "Create random logits <code>[8, 5]</code> and ids with values 0–4, compute both, and compare the two "
             "<code>Item()</code> values: they are equal (SparseCrossEntropy builds the one-hot targets on the device internally)."),
            (2, "Why is the smoothed loss (0.3455) higher than the plain one (0.1755) for the same correct predictions?",
             "With s = 0.1 the target puts 0.1/3 on each wrong class, so the confident predictions are now penalized for "
             "giving those classes almost no probability. Smoothing trades a higher training loss for calmer confidence."),
            (3, "Implement the Huber loss above and compare it with MSE and MAE on data with 5% outliers.",
             "Generate y = 2x + noise, replace 5% of targets by values 20× larger, train three identical models, and "
             "compare the fitted slope with 2. MSE is pulled furthest by the outliers; Huber and MAE stay close, with "
             "Huber usually converging more smoothly than MAE."),
        ], PART),
        footer("Loss", "MSE", "MAE", "Cross-entropy", "Sparse cross-entropy", "Binary cross-entropy", "Logits",
               "Label smoothing", "Outlier", "Huber loss"),
    )
