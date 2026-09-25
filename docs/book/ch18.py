"""Chapter 18 — The Trainer."""
from gen import *

PART = "III"


def build():
    return page(
        chapter_open(
            "trainer",
            "Every training loop in the previous chapters had the same shape: batches in, forward, loss, zero, backward, "
            "step, measure, repeat. The <code>Trainer</code> packages that loop with everything around it: metrics on "
            "the device, validation, early stopping with best-weight restore, learning-rate schedules, gradient "
            "clipping, cancellation and telemetry. This chapter shows a complete run and every option.",
            "<code>new Trainer(model, optimizer, loss)</code>, then <code>Fit(trainLoader, epochs, validationLoader)</code>.",
            "Options: <code>Metrics</code>, <code>EarlyStoppingPatience</code>, <code>RestoreBestWeights</code>, <code>MinImprovement</code>, <code>Scheduler</code>, <code>MaxGradientNorm</code>.",
            "<code>Fit</code> returns a <code>TrainingHistory</code>; <code>Evaluate</code> scores a loader; <code>Predict</code> returns a <code>float[,]</code> for a dataset.",
            "It manages modes, scopes and memory itself, and reads the GPU once per epoch.",
            "Progress output comes from telemetry: subscribe a <code>ConsoleLogger</code> (" + "{ch}" + ").",
        ).replace("{ch}", ch("telemetry")),
        h2("18.1 A complete run"),
        mex("regression with validation, a schedule and early stopping",
            "2,000 synthetic samples, y = 3·x₀ − 2·x₁² + sin(x₂) + noise, with two useless extra features.",
            """
            using NeuralSharp;
            using NeuralSharp.Data;
            using NeuralSharp.Diagnostics;
            using NeuralSharp.Layers;
            using NeuralSharp.Optimizers;
            using NeuralSharp.Training;

            var rng = new Random(7);
            var features = new float[2000, 5];
            var targets = new float[2000, 1];
            for (int i = 0; i < 2000; i++)
            {
                for (int j = 0; j < 5; j++) features[i, j] = rng.NextSingle() * 4 - 2;
                targets[i, 0] = 3 * features[i, 0] - 2 * features[i, 1] * features[i, 1]
                              + MathF.Sin(features[i, 2]) + 0.1f * (rng.NextSingle() - 0.5f);
            }
            var (train, test) = Dataset.FromArrays(features, targets).Split(0.8, seed: 1);
            var fx = StandardScaler.FitFeatures(train);
            var fy = StandardScaler.FitTargets(train);
            train = train.Scale(fx, fy);
            test = test.Scale(fx, fy);

            var r = new Random(1);
            using var model = new Sequential
            {
                new Linear(5, 64, random: r), new ReLU(),
                new Linear(64, 64, random: r), new ReLU(),
                new Linear(64, 1, random: r),
            };
            using var optimizer = new AdamW(model.Parameters(), learningRate: 3e-3f, weightDecay: 1e-4f);
            var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
            {
                Metrics = { Metric.MeanAbsoluteError },
                EarlyStoppingPatience = 15,
                Scheduler = new CosineAnnealing(optimizer, totalEpochs: 200, warmupEpochs: 3),
            };

            using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 25)))
            {
                var history = trainer.Fit(new DataLoader(train, batchSize: 64, shuffle: true, seed: 1), epochs: 200,
                                          validation: new DataLoader(test, batchSize: 256));
                Console.WriteLine($"best epoch {history.BestEpoch}, stopped early: {history.StoppedEarly}, " +
                                  $"epochs run {history.Epochs.Count}");
            }
            """,
            out="""
            Sequential(5 layers)
              Linear(5 -> 64)  [384 params]
              ReLU
              Linear(64 -> 64)  [4,160 params]
              ReLU
              Linear(64 -> 1)  [65 params]
            Total trainable parameters: 4,609
            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 1,600 samples, 400 validation | batch 64, 25 steps/epoch | AdamW lr=0.00075 | 4,609 parameters | 4 CPU threads
            Epoch   1/200  loss 0.585848  mae 0.6183  val_loss 0.319308  val_mae 0.4663  70.6 ms  22,674 samples/s  *
            Epoch  25/200  loss 0.002065  mae 0.0352  val_loss 0.003152  val_mae 0.0438  21.0 ms  76,312 samples/s
            Epoch  50/200  loss 0.000613  mae 0.0197  val_loss 0.001295  val_mae 0.0280  11.7 ms  137,020 samples/s
            Epoch  75/200  loss 0.000412  mae 0.0161  val_loss 0.000995  val_mae 0.0244  6.9 ms  230,481 samples/s
            Epoch 100/200  loss 0.000265  mae 0.0129  val_loss 0.000840  val_mae 0.0228  4.6 ms  350,002 samples/s  *
            Epoch 125/200  loss 0.000222  mae 0.0118  val_loss 0.000764  val_mae 0.0218  4.0 ms  399,341 samples/s
            Finished 135 epochs in 1.46 s (early stop) | best epoch 120 loss 0.000743
            best epoch 120, stopped early: True, epochs run 135
            """),
        para("The starting rate reads 0.00075 because the warm-up begins at a quarter of 3e-3. A star marks a new "
             "best validation loss. Early stopping ended the run 15 epochs after the best epoch (120) and restored "
             "that epoch's weights. Epoch times fall as .NET's JIT compiler optimizes the hot code; the first "
             "epoch also pays one-off start-up costs."),
        mex("scoring the result in real units", None,
            """
            var result = trainer.Evaluate(new DataLoader(test, batchSize: 256));
            Console.WriteLine($"test loss {result.Loss:F5}, mae {result.Metrics["mae"]:F4} (scaled units), {result.Samples} samples");

            float[,] predicted = trainer.Predict(test);                 // [400, 1], scaled
            var p = new float[test.Count];
            var a = new float[test.Count];
            for (int i = 0; i < test.Count; i++) { p[i] = predicted[i, 0]; a[i] = test.GetTargets(i)[0]; }
            fy.InverseTransform(p, 1);
            fy.InverseTransform(a, 1);
            var report = RegressionReport.Compute(p, a);
            Console.WriteLine($"MAE {report.MeanAbsoluteError:F3}  RMSE {report.RootMeanSquaredError:F3}  R² {report.RSquared:F4}");
            """,
            out="""
            test loss 0.00074, mae 0.0213 (scaled units), 400 samples
            MAE 0.090  RMSE 0.116  R² 0.9992
            """),
        cpugpu("the same run on either device",
               """
               Device.Default = Device.Cpu;        // before the model and loaders are created
               """,
               """
               Device.Default = Device.Cuda();
               // Model, loaders and optimizer state now live on the GPU. The loss and metrics of
               // every batch are summed in a device tensor; the host reads it ONCE per epoch,
               // so the GPU never waits for the CPU inside an epoch.
               """,
               "At 4,609 parameters and batches of 64 the CPU is the right choice; " + ch("performance") + " explains when the GPU wins."),
        h2("18.2 Options"),
        reftable(["Property", "Default", "Effect"], [
            ["<code>Metrics</code>", "empty", "Extra quantities per epoch, for training and validation (" + ch("metrics") + ")"],
            ["<code>EarlyStoppingPatience</code>", "null (off)", "Stop after this many epochs without improvement of the monitored loss"],
            ["<code>RestoreBestWeights</code>", "true", "On early stop, load the best epoch's weights back"],
            ["<code>MinImprovement</code>", "0", "Decrease needed to count as an improvement"],
            ["<code>Scheduler</code>", "null", "Stepped after every epoch (" + ch("schedules") + ")"],
            ["<code>MaxGradientNorm</code>", "null", "Clip gradients to this global norm before each step"],
        ], caption="Table 18.1 — Trainer options"),
        reftable(["Method", "Returns"], [
            ["<code>Fit(train, epochs, validation?, cancellationToken?)</code>", "<code>TrainingHistory</code>: <code>Epochs</code> (one <code>EpochCompleted</code> each), <code>BestEpoch</code>, <code>BestLoss</code>, <code>StoppedEarly</code>"],
            ["<code>Evaluate(loader)</code>", "<code>EvaluationResult(Loss, Metrics, Samples)</code>, in evaluation mode without gradients"],
            ["<code>Predict(dataset, batchSize = 1024)</code>", "<code>float[Count, outputs]</code>, batch by batch"],
        ], caption="Table 18.2 — Trainer methods"),
        deriv("What Fit does in each epoch", [
            "Switches the model to training mode and resets the device-side sums.",
            "For each batch: opens a <code>TensorScope</code>; runs forward, loss, <code>ZeroGrad</code>, <code>Backward</code>; clips if "
            "<code>MaxGradientNorm</code> is set; <code>Step</code>; adds loss and metrics (weighted by batch size) to the sums; disposes the batch.",
            "Reads the sums once and evaluates the validation loader (evaluation mode, no gradients).",
            "Compares the monitored loss (validation if given, else training) with the best so far; keeps a copy of the best weights when early stopping is on.",
            "Publishes an <code>EpochCompleted</code> event, steps the scheduler, and stops if patience is exhausted.",
        ]),
        h2("18.3 Cancelling and running in the background"),
        snippet("""
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(10));   // or a Stop button
            try
            {
                var history = await Task.Run(() => trainer.Fit(loader, epochs: 1000, validation: val, cancellationToken: cts.Token));
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("stopped; the model keeps the weights of the last completed epoch");
                model.Save("partial.weights");
            }
            """, caption="Cancellation is checked between epochs"),
        para("Running <code>Fit</code> inside <code>Task.Run</code> keeps a UI or web server responsive; the whole "
             "fit runs on one thread, so <code>TensorScope</code>s work as usual (" + ch("memory") + ")."),
        h2("18.4 When to write your own loop"),
        reftable(["Need", "Trainer?"], [
            ["Standard supervised learning: inputs → targets", "Yes"],
            ["Several optimizers or parameter groups (" + ch("optimizers") + ")", "Own loop"],
            ["Several inputs per sample, or losses combining several model outputs", "Own loop (or pack the inputs into one tensor and split with <code>Narrow</code>)"],
            ["Gradient accumulation (" + ch("autograd") + ")", "Own loop"],
            ["Unsupervised (autoencoder): targets = inputs", "Yes: build the dataset with the features as targets (" + ch("anomaly") + ")"],
        ], caption="Table 18.3 — Trainer or hand-written loop"),
        snippet("""
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                model.Train();
                foreach (var batch in trainLoader)
                {
                    using var _ = batch;
                    using var scope = new TensorScope();
                    var loss = MyLoss(model.Forward(batch.Features), batch.Targets);
                    foreach (var o in optimizers) o.ZeroGrad();
                    loss.Backward();
                    foreach (var o in optimizers) o.Step();
                }
                using (Autograd.NoGrad()) { model.Eval(); /* validation, logging */ }
            }
            """, caption="The skeleton of a hand-written loop"),
        trap("a validation set that overlaps the training set",
             "<p>If the same rows (or near-duplicates, such as overlapping time windows) are in both, the validation "
             "loss looks excellent and early stopping trusts it. Split before building windows, and for time series "
             "split by time, not randomly (" + ch("timeseries") + ").</p>"),
        trap("early stopping on the training loss",
             "<p>Without a validation loader, early stopping watches the training loss, which usually keeps falling; it "
             "then only ends runs that have stopped learning altogether. Pass a validation loader.</p>"),
        practice([
            (1, "Add accuracy to a classification Trainer and stop after 10 epochs without validation improvement.",
             "<code>new Trainer(model, opt, Losses.SparseCrossEntropy) { Metrics = { Metric.SparseAccuracy }, EarlyStoppingPatience = 10 }</code> "
             "and pass <code>validation:</code> to <code>Fit</code>."),
            (1, "Where do you find the validation MAE of epoch 42 after training?",
             "<code>history.Epochs[41].ValidationMetrics![\"mae\"]</code> (epochs are 1-based, the list 0-based)."),
            (2, "Why can the Trainer train on the GPU without a synchronization per batch, while your hand-written loop that prints <code>loss.Item()</code> every step cannot?",
             "The Trainer adds each batch's loss into a device tensor with a kernel and reads the totals once per epoch. "
             "<code>Item()</code> copies to the host and waits for all queued work every time it is called."),
            (2, "Save the best model to disk during training, not only at the end.",
             "Subscribe a telemetry hook whose <code>OnEpochCompleted</code> calls <code>model.Save(path)</code> when "
             "<code>e.IsBest</code> is true (" + ch("telemetry") + ")."),
            (3, "Train the Section 18.1 model with MAE as the loss instead of MSE, and compare the final real-unit MAE and RMSE.",
             "Pass <code>Losses.MeanAbsoluteError</code> as the loss (keep MAE as a metric). Measured: early stopping at "
             "epoch 88 (best 73), MAE 0.085 and RMSE 0.106, against 0.090 and 0.116 with MSE. Both scores improved here; on "
             "clean data without outliers the choice matters little, and a different seed can reverse small differences."),
        ], PART),
        footer("Trainer", "Epoch", "Validation set", "Early stopping", "Patience", "Training history",
               "Evaluation", "Cancellation"),
    )
