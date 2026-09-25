using System.Diagnostics;
using NeuralSharp.Data;
using NeuralSharp.Diagnostics;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;

namespace NeuralSharp.Training;

/// <summary>A quantity tracked during training and evaluation, computed as a mean over samples.</summary>
/// <param name="Name">Short name used in logs, e.g. "mae".</param>
/// <param name="BatchMean">Returns the batch mean as a scalar tensor, given (predictions, targets).</param>
/// <param name="Finalize">Optional transform of the epoch mean, e.g. a square root for RMSE.</param>
public sealed record Metric(string Name, Func<Tensor, Tensor, Tensor> BatchMean, Func<double, double>? Finalize = null)
{
    /// <summary>Mean absolute error.</summary>
    public static Metric MeanAbsoluteError { get; } = new("mae", Losses.MeanAbsoluteError);

    /// <summary>Mean squared error.</summary>
    public static Metric MeanSquaredError { get; } = new("mse", Losses.MeanSquaredError);

    /// <summary>Root mean squared error (in the units of the targets).</summary>
    public static Metric RootMeanSquaredError { get; } = new("rmse", Losses.MeanSquaredError, Math.Sqrt);

    /// <summary>
    /// Classification accuracy: the arg-max of each prediction row matches the arg-max of the one-hot target.
    /// For single-column outputs it compares probabilities with 0.5 (see <see cref="BinaryAccuracy"/>).
    /// </summary>
    public static Metric Accuracy { get; } = new("accuracy", (p, t) => Tensor.MatchRate(p, t, 0.5f));

    /// <summary>Accuracy with integer class targets (the logits' shape without the last dimension), e.g. next-token prediction.</summary>
    public static Metric SparseAccuracy { get; } = new("accuracy", (p, t) =>
    {
        using var oneHot = Tensor.OneHot(t.Reshape(p.Shape[..^1]), p.Shape[^1]);
        return Tensor.MatchRate(p, oneHot, 0.5f);
    });

    /// <summary>Binary accuracy for a single output column: (prediction ≥ threshold) == (target ≥ 0.5). Use threshold 0 for logits.</summary>
    public static Metric BinaryAccuracy(float threshold = 0.5f) => new("accuracy", (p, t) => Tensor.MatchRate(p, t, threshold));
}

/// <summary>Loss and metrics of a model on a dataset.</summary>
/// <param name="Loss">Mean loss.</param>
/// <param name="Metrics">Metric values by name.</param>
/// <param name="Samples">Samples evaluated.</param>
public sealed record EvaluationResult(double Loss, IReadOnlyDictionary<string, double> Metrics, int Samples);

/// <summary>Per-epoch results of <see cref="Trainer.Fit"/>.</summary>
public sealed class TrainingHistory
{
    internal List<EpochCompleted> EpochList { get; } = [];

    /// <summary>One entry per completed epoch.</summary>
    public IReadOnlyList<EpochCompleted> Epochs => EpochList;

    /// <summary>The epoch with the best monitored loss (1-based).</summary>
    public int BestEpoch { get; internal set; }

    /// <summary>The best monitored loss (validation loss when a validation set was used).</summary>
    public double BestLoss { get; internal set; } = double.PositiveInfinity;

    /// <summary>Whether early stopping ended training.</summary>
    public bool StoppedEarly { get; internal set; }
}

/// <summary>
/// Runs the training loop: batches from a <see cref="DataLoader"/>, forward, loss, backward, optimizer
/// step, metrics, validation, early stopping, and telemetry for every step. Memory for each step is
/// recycled automatically, and the loss is accumulated on the device so the GPU is only synchronized
/// once per epoch (unless per-batch telemetry is requested).
/// </summary>
/// <example>
/// <code>
/// var trainer = new Trainer(model, new Adam(model.Parameters(), 1e-3f), Losses.MeanSquaredError)
/// {
///     Metrics = { Metric.MeanAbsoluteError },
///     EarlyStoppingPatience = 20,
/// };
/// var history = trainer.Fit(trainLoader, epochs: 300, validation: testLoader);
/// </code>
/// </example>
public sealed class Trainer(Module model, Optimizer optimizer, Func<Tensor, Tensor, Tensor> loss)
{
    /// <summary>The model being trained.</summary>
    public Module Model { get; } = model;

    /// <summary>The optimizer updating the model.</summary>
    public Optimizer Optimizer { get; } = optimizer;

    /// <summary>The loss function (predictions, targets) → scalar.</summary>
    public Func<Tensor, Tensor, Tensor> Loss { get; } = loss;

    /// <summary>Extra quantities to report each epoch.</summary>
    public List<Metric> Metrics { get; } = [];

    /// <summary>
    /// Stop when the monitored loss (validation loss if available, else training loss) has not improved
    /// for this many epochs. Null disables early stopping.
    /// </summary>
    public int? EarlyStoppingPatience { get; init; }

    /// <summary>When early stopping is on, restore the weights of the best epoch at the end. Default true.</summary>
    public bool RestoreBestWeights { get; init; } = true;

    /// <summary>Minimum decrease of the monitored loss that counts as an improvement.</summary>
    public double MinImprovement { get; init; }

    /// <summary>Adjusts the learning rate after every epoch (e.g. <see cref="CosineAnnealing"/>).</summary>
    public LearningRateScheduler? Scheduler { get; init; }

    /// <summary>When set, gradients are clipped to this global L2 norm before each step (recommended for RNNs).</summary>
    public float? MaxGradientNorm { get; init; }

    private Device Device => Model.Parameters().FirstOrDefault()?.Device ?? Device.Default;

    /// <summary>Trains for up to <paramref name="epochs"/> passes over <paramref name="train"/>.</summary>
    public TrainingHistory Fit(DataLoader train, int epochs, DataLoader? validation = null, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        var device = Device;
        var history = new TrainingHistory();
        var totalTime = Stopwatch.StartNew();
        float[][]? bestWeights = null;
        int epochsWithoutImprovement = 0;
        long step = 0;
        bool cancelled = false;

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingStarted(new TrainingStarted(
                Model.Summary(), Optimizer.GetType().Name, device, epochs, train.SampleCount, validation?.SampleCount,
                train.BatchSize, train.BatchCount, Model.ParameterCount, Optimizer.LearningRate, ComputeResources.MaxCpuThreads));
        }

        // Running sums live on the device: [loss, metric 1, metric 2, ...], each weighted by batch size.
        using var sums = Tensor.Zeros([1 + Metrics.Count], device);
        int epoch = 0;
        for (epoch = 1; epoch <= epochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                epoch--;
                break;
            }

            var epochTime = Stopwatch.StartNew();
            Model.Train();
            sums.Backend.Fill(sums.Storage, sums.Size, 0f);
            int samples = 0;
            bool perBatch = Telemetry.IsEnabled(TelemetryLevel.Batches);
            long dataStart = Stopwatch.GetTimestamp();

            foreach (var batch in train)
            {
                using var _ = batch;
                using var scope = new TensorScope();
                var dataTime = Stopwatch.GetElapsedTime(dataStart);
                long computeStart = Stopwatch.GetTimestamp();

                var predictions = Model.Forward(batch.Features);
                var batchLoss = Loss(predictions, batch.Targets);
                Optimizer.ZeroGrad();
                batchLoss.Backward();
                double? gradientNorm = MaxGradientNorm is { } maxNorm ? Optimizer.ClipGradientNorm(maxNorm)
                    : Telemetry.IsEnabled(TelemetryLevel.Gradients) ? Optimizer.GradientNorm() : null;
                Optimizer.Step();
                Accumulate(sums, batchLoss, predictions, batch.Targets, batch.Size);
                samples += batch.Size;
                step++;

                if (perBatch)
                {
                    double lossValue = batchLoss.Item(); // synchronizes, so compute time below is real time
                    Telemetry.BatchCompleted(new BatchCompleted(epoch, batch.Index + 1, train.BatchCount, step, batch.Size,
                        lossValue, Optimizer.LearningRate, gradientNorm, dataTime, Stopwatch.GetElapsedTime(computeStart)));
                }

                dataStart = Stopwatch.GetTimestamp();
            }

            var (trainLoss, trainMetrics) = Read(sums, samples);
            EvaluationResult? validationResult = validation is null ? null : Evaluate(validation);
            double monitored = validationResult?.Loss ?? trainLoss;
            bool isBest = monitored < history.BestLoss - MinImprovement;
            if (isBest)
            {
                history.BestLoss = monitored;
                history.BestEpoch = epoch;
                epochsWithoutImprovement = 0;
                if (EarlyStoppingPatience is not null && RestoreBestWeights)
                {
                    bestWeights = [.. Model.Parameters().Select(p => p.ToArray())];
                }
            }
            else
            {
                epochsWithoutImprovement++;
            }

            epochTime.Stop();
            var summary = new EpochCompleted(epoch, epochs, trainLoss, trainMetrics, validationResult?.Loss, validationResult?.Metrics,
                Optimizer.LearningRate, epochTime.Elapsed, samples / Math.Max(epochTime.Elapsed.TotalSeconds, 1e-9),
                ComputeResources.GetMemoryUsage(device), isBest);
            history.EpochList.Add(summary);
            if (Telemetry.IsEnabled(TelemetryLevel.Training))
            {
                Telemetry.EpochCompleted(summary);
            }

            Scheduler?.Step();
            if (EarlyStoppingPatience is { } patience && epochsWithoutImprovement >= patience)
            {
                history.StoppedEarly = true;
                break;
            }
        }

        if (bestWeights is not null && history.StoppedEarly)
        {
            foreach (var (parameter, values) in Model.Parameters().Zip(bestWeights))
            {
                parameter.Load(values);
            }
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(Math.Min(epoch, epochs), totalTime.Elapsed,
                history.Epochs.Count > 0 ? history.Epochs[^1].Loss : double.NaN, history.BestEpoch, history.BestLoss, history.StoppedEarly, cancelled));
        }

        cancellationToken.ThrowIfCancellationRequested();
        return history;
    }

    /// <summary>Computes the loss and metrics on <paramref name="data"/> in evaluation mode.</summary>
    public EvaluationResult Evaluate(DataLoader data)
    {
        bool wasTraining = Model.IsTraining;
        Model.Eval();
        try
        {
            using var sums = Tensor.Zeros([1 + Metrics.Count], data.Device);
            int samples = 0;
            using (Autograd.NoGrad())
            {
                foreach (var batch in data)
                {
                    using var _ = batch;
                    using var scope = new TensorScope();
                    var predictions = Model.Forward(batch.Features);
                    Accumulate(sums, Loss(predictions, batch.Targets), predictions, batch.Targets, batch.Size);
                    samples += batch.Size;
                }
            }

            var (lossValue, metrics) = Read(sums, samples);
            return new EvaluationResult(lossValue, metrics, samples);
        }
        finally
        {
            Model.Train(wasTraining);
        }
    }

    /// <summary>Predicts every sample of <paramref name="data"/>, batch by batch; returns [Count, outputs] (each sample's output flattened).</summary>
    public float[,] Predict(Dataset data, int batchSize = 1024)
    {
        var device = Device;
        float[,]? result = null;
        foreach (var batch in new DataLoader(data, batchSize, device: device))
        {
            using var _ = batch;
            using var output = Model.Predict(batch.Features);
            var values = output.ToArray();
            int outputs = output.Size / batch.Size;
            result ??= new float[data.Count, outputs];
            int row0 = batch.Index * batchSize;
            for (int r = 0; r < batch.Size; r++)
            {
                for (int c = 0; c < outputs; c++)
                {
                    result[row0 + r, c] = values[r * outputs + c];
                }
            }
        }

        return result ?? new float[0, 0];
    }

    /// <summary>Adds batch-size-weighted loss and metric means into the device-side running sums.</summary>
    private void Accumulate(Tensor sums, Tensor batchLoss, Tensor predictions, Tensor targets, int batchSize)
    {
        var backend = sums.Backend;
        backend.AxpyAt(batchLoss.Storage, sums.Storage, 0, batchSize);
        if (Metrics.Count == 0)
        {
            return;
        }

        using (Autograd.NoGrad())
        {
            for (int i = 0; i < Metrics.Count; i++)
            {
                using var value = Metrics[i].BatchMean(predictions, targets);
                backend.AxpyAt(value.Storage, sums.Storage, i + 1, batchSize);
            }
        }
    }

    private (double Loss, Dictionary<string, double> Metrics) Read(Tensor sums, int samples)
    {
        var values = sums.ToArray();
        double n = Math.Max(samples, 1);
        var metrics = new Dictionary<string, double>(Metrics.Count);
        for (int i = 0; i < Metrics.Count; i++)
        {
            double mean = values[i + 1] / n;
            metrics[Metrics[i].Name] = Metrics[i].Finalize?.Invoke(mean) ?? mean;
        }

        return (values[0] / n, metrics);
    }
}

/// <summary>Standard regression scores computed from predictions and true values.</summary>
/// <param name="MeanAbsoluteError">Mean |prediction - actual|.</param>
/// <param name="RootMeanSquaredError">sqrt(mean((prediction - actual)²)).</param>
/// <param name="MeanAbsolutePercentageError">Mean |prediction - actual| / |actual|, as a fraction (0.05 = 5%).</param>
/// <param name="RSquared">Coefficient of determination: 1 is perfect, 0 is no better than predicting the mean.</param>
public readonly record struct RegressionReport(double MeanAbsoluteError, double RootMeanSquaredError, double MeanAbsolutePercentageError, double RSquared)
{
    /// <summary>Compares predictions with actual values (same length, any column layout).</summary>
    public static RegressionReport Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> actual)
    {
        if (predicted.Length != actual.Length || actual.Length == 0)
        {
            throw new ArgumentException("predicted and actual must be non-empty and the same length.");
        }

        double mean = 0;
        foreach (float a in actual)
        {
            mean += a;
        }

        mean /= actual.Length;
        double abs = 0, sq = 0, pct = 0, total = 0;
        int pctCount = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double error = predicted[i] - actual[i];
            abs += Math.Abs(error);
            sq += error * error;
            total += (actual[i] - mean) * (actual[i] - mean);
            if (actual[i] != 0)
            {
                pct += Math.Abs(error / actual[i]);
                pctCount++;
            }
        }

        return new RegressionReport(abs / actual.Length, Math.Sqrt(sq / actual.Length), pctCount > 0 ? pct / pctCount : double.NaN,
            total > 0 ? 1 - sq / total : double.NaN);
    }
}
