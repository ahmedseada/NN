"""Chapter 14 — Metrics."""
from gen import *

PART = "III"


def build():
    return page(
        chapter_open(
            "metrics",
            "The loss is what training optimizes; metrics are what you and your users care about: the percentage of "
            "correct answers, the average error in dollars, how much of the variation the model explains. NeuralSharp "
            "tracks metrics on the device during training (so they cost no extra synchronization), and computes "
            "standard regression scores on plain arrays after training. This chapter covers both and shows how to add "
            "your own.",
            "Built-in <code>Metric</code>s: <code>MeanAbsoluteError</code>, <code>MeanSquaredError</code>, <code>RootMeanSquaredError</code>, <code>Accuracy</code>, <code>SparseAccuracy</code>, <code>BinaryAccuracy(threshold)</code>.",
            "Add them to <code>trainer.Metrics</code>; they appear per epoch for training and validation data.",
            "<code>RegressionReport.Compute(predicted, actual)</code> gives MAE, RMSE, MAPE and R² in real units.",
            "A custom metric is a name plus a function returning the batch mean as a scalar tensor.",
            "Report metrics on unscaled values and on data the model never trained on.",
        ),
        h2("14.1 Metrics during training"),
        reftable(["Metric", "Name in logs", "Predictions / targets", "Meaning"], [
            ["<code>Metric.MeanAbsoluteError</code>", "mae", "any / same shape", "mean |p − t|"],
            ["<code>Metric.MeanSquaredError</code>", "mse", "any / same shape", "mean (p − t)²"],
            ["<code>Metric.RootMeanSquaredError</code>", "rmse", "any / same shape", "√MSE of the whole epoch, in target units"],
            ["<code>Metric.Accuracy</code>", "accuracy", "<code>[N, K]</code> / one-hot <code>[N, K]</code>", "ArgMax matches (for one column: p ≥ 0.5 vs t ≥ 0.5)"],
            ["<code>Metric.SparseAccuracy</code>", "accuracy", "<code>[..., K]</code> / ids <code>[...]</code>", "ArgMax equals the id (per token for sequences)"],
            ["<code>Metric.BinaryAccuracy(threshold)</code>", "accuracy", "<code>[N, 1]</code> / 0 or 1", "(p ≥ threshold) equals (t ≥ 0.5); use threshold 0 for logits"],
        ], caption="Table 14.1 — Built-in metrics (NeuralSharp.Training.Metric)"),
        mex("the classification metrics on a small batch", None,
            """
            var logits = Tensor.From(new float[,] {
                { 2.0f, 0.5f, -1.0f }, { 0.1f, 0.2f, 3.0f }, { 1.0f, 1.5f, 0.2f }, { 0.3f, 0.1f, 0.0f } });
            var oneHot = Tensor.From(new float[,] { { 1, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 1, 0, 0 } });
            var ids = Tensor.From([0f, 2f, 0f, 0f]);
            Console.WriteLine($"Accuracy {Metric.Accuracy.BatchMean(logits, oneHot).Item():F2}  " +
                              $"SparseAccuracy {Metric.SparseAccuracy.BatchMean(logits, ids).Item():F2}");

            var probabilities = Tensor.From(new float[,] { { 0.9f }, { 0.4f }, { 0.6f }, { 0.2f } });
            var labels = Tensor.From(new float[,] { { 1 }, { 0 }, { 0 }, { 0 } });
            Console.WriteLine($"BinaryAccuracy(0.5) {Metric.BinaryAccuracy().BatchMean(probabilities, labels).Item():F2}  " +
                              $"BinaryAccuracy(0.7) {Metric.BinaryAccuracy(0.7f).BatchMean(probabilities, labels).Item():F2}");
            """,
            out="""
            Accuracy 0.75  SparseAccuracy 0.75
            BinaryAccuracy(0.5) 0.75  BinaryAccuracy(0.7) 1.00
            """,
            after="Row 3 predicts class 1 but the answer is 0, hence 3 of 4. Raising the yes/no threshold to 0.7 turns "
                  "the false positive (0.6 for a 0) into a correct no; " + ch("binary") + " shows how to pick a threshold."),
        cpugpu("attaching metrics to the Trainer",
               """
               var trainer = new Trainer(model, optimizer, Losses.SparseCrossEntropy)
               {
                   Metrics = { Metric.SparseAccuracy },
               };
               // log line: ... loss 0.412  accuracy 0.8710  val_loss 0.455  val_accuracy 0.8590
               """,
               """
               // identical on the GPU: each batch adds its metric into a small device tensor,
               // and the epoch reads everything back in ONE synchronization
               """),
        h2("14.2 Regression scores after training"),
        para("<code>RegressionReport.Compute(predicted, actual)</code> works on plain <code>float</code> spans, so "
             "apply it after unscaling predictions back to real units (" + ch("data") + ")."),
        mex("four house prices", None,
            """
            var report = RegressionReport.Compute(
                predicted: [310_000f, 205_000f, 150_000f, 480_000f],
                actual:    [300_000f, 220_000f, 140_000f, 500_000f]);
            Console.WriteLine($"MAE {report.MeanAbsoluteError:N0}  RMSE {report.RootMeanSquaredError:N0}  " +
                              $"MAPE {report.MeanAbsolutePercentageError:P1}  R² {report.RSquared:F3}");
            """,
            out="""
            MAE 13,750  RMSE 14,361  MAPE 5.3 %  R² 0.988
            """),
        reftable(["Score", "Reads as", "Good for"], [
            ["MAE", "average miss, in target units ($13,750)", "explaining to people"],
            ["RMSE", "like MAE but weights large misses more (≥ MAE)", "spotting occasional big errors"],
            ["MAPE", "average miss as a share of the true value (5.3%)", "comparing across scales; undefined for true values of 0 (skipped)"],
            ["R²", "share of the targets' variation explained: 1 perfect, 0 = predicting the mean, negative = worse", "overall fit (glossary <b>R²</b>)"],
        ], caption="Table 14.2 — The regression report"),
        h2("14.3 Custom metrics"),
        para("A <code>Metric</code> is a record: a name, a function <code>(predictions, targets) =&gt; batch mean as a "
             "scalar tensor</code>, and an optional <code>Finalize</code> applied to the epoch mean (RMSE uses "
             "<code>Math.Sqrt</code>). The Trainer weights each batch mean by the batch size, so the epoch value is an "
             "exact mean over samples."),
        snippet("""
            // Stays on the device: mean signed error (does the model over- or under-predict?)
            var bias = new Metric("bias", (p, t) => (p - t).Mean());

            // Root mean squared log error, for targets that span orders of magnitude (targets > 0)
            var rmsle = new Metric("rmsle",
                (p, t) => ((p.Relu() + 1f).Log() - (t + 1f).Log()).Square().Mean(),
                Math.Sqrt);

            var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
            {
                Metrics = { Metric.MeanAbsoluteError, bias, rmsle },
            };
            """, caption="Custom metrics built from tensor operations"),
        para("For the four prices above, <code>bias</code> is (10,000 − 15,000 + 10,000 − 20,000) / 4 = −3,750: "
             "the model under-predicts slightly on average."),
        snippet("""
            // Needs host logic: share of predictions within 10% of the truth.
            var within10 = new Metric("within10%", (p, t) =>
            {
                float[] pv = p.ToArray(), tv = t.ToArray();          // synchronizes the device
                int hits = 0;
                for (int i = 0; i < pv.Length; i++)
                    if (MathF.Abs(pv[i] - tv[i]) <= 0.1f * MathF.Abs(tv[i])) hits++;
                return Tensor.Scalar(hits / (float)pv.Length, p.Device);
            });
            """, caption="A metric computed on the host (simple, but slower on the GPU)"),
        trap("metrics that read data back on every batch",
             "<p>A metric that calls <code>ToArray()</code> or <code>Item()</code> forces the GPU to finish all queued "
             "work on every batch. It gives the right numbers but can make GPU training several times slower. Prefer "
             "tensor-only metrics, or compute host-side scores once after training with "
             "<code>trainer.Predict</code> (" + ch("trainer") + ").</p>"),
        trap("accuracy on imbalanced data",
             "<p>If 95% of emails are not spam, a model that always says \"not spam\" is 95% accurate. Look at the "
             "accuracy of each class, or at precision and recall (" + ch("binary") + ").</p>"),
        practice([
            (1, "Which metric would you show a user of a house-price model, and in what units?",
             "MAE in currency on unscaled predictions (e.g. \"typically within $13,750\"), perhaps with MAPE (\"5.3%\")."),
            (1, "A classifier reports <code>accuracy 0.99</code> on training data and <code>val_accuracy 0.71</code>. What is happening?",
             "Overfitting: the model memorizes the training set. Add regularization (dropout, weight decay), more data, or "
             "stop earlier (" + ch("norm") + ", " + ch("trainer") + ")."),
            (2, "Write a tensor-only metric for the mean of the predictions (useful to watch for collapse to a constant).",
             "<code>new Metric(\"pred_mean\", (p, t) =&gt; p.Mean())</code>."),
            (2, "Compute R² by hand for predictions equal to the mean of the targets.",
             "The squared error equals the total variation, so R² = 1 − 1 = 0."),
            (3, "Add per-class accuracy for a 3-class problem as three custom metrics.",
             "For class c, build a mask from the one-hot targets, <code>m = t.Narrow(1, c, 1)</code>, and the correctness "
             "per row, <code>(Tensor.OneHot(p.ArgMax(), 3) * t).Sum(1, keepDim: true)</code>. The metric is "
             "sum(m · correct) / sum(m); because a Metric returns a batch mean, return "
             "<code>(m * correct).Mean()</code> and divide by the class frequency afterwards, or compute it after training "
             "from <code>trainer.Predict</code>."),
        ], PART),
        footer("Metric", "Accuracy", "MAE", "RMSE", "MAPE", "R²", "Threshold", "Class imbalance", "Validation set"),
    )
