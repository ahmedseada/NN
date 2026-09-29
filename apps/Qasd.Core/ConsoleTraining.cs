using System.Diagnostics;
using NeuralSharp.Datasets;

namespace Qasd;

/// <summary>Training and scoring with a progress bar (NeuralSharp's <see cref="ConsoleStatus"/>), for the command line.</summary>
public static class ConsoleTraining
{
    /// <summary>
    /// Trains a <see cref="TextClassifier"/> with a bar over all the optimizer steps; <paramref name="epochLines"/> writes
    /// each epoch's loss and validation accuracy above it.
    /// </summary>
    public static TextClassifier Classifier(IReadOnlyList<LabeledText> train, TextClassifierOptions options, bool epochLines = true,
        string label = "training", Action<TextClassifierEpoch>? onEpoch = null)
    {
        var status = new ConsoleStatus();
        var clock = Stopwatch.StartNew();
        int epochs = Math.Max(1, options.Epochs), lastEpoch = 0, lastStep = 0;
        var classifier = TextClassifier.Train(train, options,
            epoch =>
            {
                lastEpoch = epoch.Epoch;
                onEpoch?.Invoke(epoch);
                if (epochLines)
                {
                    status.Log($"  epoch {epoch.Epoch,3}: loss {epoch.Loss:F4}"
                               + (double.IsNaN(epoch.ValidationAccuracy) ? "" : $", validation accuracy {epoch.ValidationAccuracy:P1}")
                               + (epoch.Best ? "  *" : ""));
                }
            },
            step =>
            {
                lastStep = step.Step;
                status.Progress(label, step.Step, step.TotalSteps, clock.Elapsed, $"epoch {step.Epoch}/{epochs}, loss {step.Loss:F4}");
            });

        // Stopped early: the bar shows the steps that ran as the whole.
        status.Progress(label, lastStep, lastStep, clock.Elapsed,
            lastEpoch < epochs ? $"stopped early after epoch {lastEpoch}/{epochs} (no better validation)" : $"epoch {lastEpoch}/{epochs}");
        status.Finish();
        return classifier;
    }

    /// <summary>A progress callback (done, total) that shows a scoring bar, kept on screen when done.</summary>
    public static Action<int, int> Scoring(string label = "scoring")
    {
        var status = new ConsoleStatus();
        var clock = Stopwatch.StartNew();
        return (done, total) =>
        {
            status.Progress(label, done, total, clock.Elapsed, $"{done / Math.Max(1e-9, clock.Elapsed.TotalSeconds):F1} messages/s", "messages");
            if (done >= total)
            {
                status.Finish();
            }
        };
    }
}
