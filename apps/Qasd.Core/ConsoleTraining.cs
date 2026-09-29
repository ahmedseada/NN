using NeuralSharp.Pretrained;

namespace Qasd;

/// <summary>Training and scoring with a <see cref="ConsoleProgress"/> bar, for the command-line tools.</summary>
public static class ConsoleTraining
{
    /// <summary>
    /// Trains a <see cref="TextClassifier"/> with a bar over all the optimizer steps; <paramref name="epochLines"/> writes
    /// each epoch's loss and validation accuracy above it.
    /// </summary>
    public static TextClassifier Classifier(IReadOnlyList<LabeledText> train, TextClassifierOptions options, bool epochLines = true,
        string label = "training", Action<TextClassifierEpoch>? onEpoch = null)
    {
        ConsoleProgress? bar = null;
        int epochs = Math.Max(1, options.Epochs), lastEpoch = 0;
        var classifier = TextClassifier.Train(train, options,
            epoch =>
            {
                lastEpoch = epoch.Epoch;
                onEpoch?.Invoke(epoch);
                if (epochLines)
                {
                    bar?.WriteLine($"  epoch {epoch.Epoch,3}: loss {epoch.Loss:F4}"
                                   + (double.IsNaN(epoch.ValidationAccuracy) ? "" : $", validation accuracy {epoch.ValidationAccuracy:P1}")
                                   + (epoch.Best ? "  *" : ""));
                }
            },
            step =>
            {
                bar ??= new ConsoleProgress(label, step.TotalSteps);
                bar.Report(step.Step, $"epoch {step.Epoch}/{epochs}, loss {step.Loss:F4}");
            });
        bar?.Complete(lastEpoch < epochs ? $"stopped early after epoch {lastEpoch}/{epochs} (no better validation)" : $"epoch {lastEpoch}/{epochs}");
        return classifier;
    }

    /// <summary>Tunes a <see cref="TunedClassifier"/> with a bar over the optimizer steps.</summary>
    public static TunedClassifier Tuned(IReadOnlyList<LabeledText> train, string outputFolder, TunedOptions options, string label = "tuning")
    {
        ConsoleProgress? bar = null;
        var tuned = TunedClassifier.Train(train, outputFolder, options,
            line =>
            {
                if (bar is null)
                {
                    Console.WriteLine($"  {line}");
                }
                else
                {
                    bar.WriteLine($"  {line}");
                }
            },
            step =>
            {
                bar ??= new ConsoleProgress(label, step.TotalSteps);
                bar.Report(step.Step, $"epoch {step.Epoch}/{options.Epochs}, loss {step.Loss:F4}, {step.TokensPerSecond:N0} tokens/s");
            });
        bar?.Complete();
        return tuned;
    }

    /// <summary>A progress callback (done, total) that draws a scoring bar and completes it at the end.</summary>
    public static Action<int, int> Scoring(string label = "scoring")
    {
        ConsoleProgress? bar = null;
        var clock = System.Diagnostics.Stopwatch.StartNew();          // from the call, so the first batch counts in the rate
        return (done, total) =>
        {
            bar ??= new ConsoleProgress(label, total, "messages");
            string rate = $"{done / Math.Max(1e-9, clock.Elapsed.TotalSeconds):F1} messages/s";
            if (done >= total)
            {
                bar.Complete(rate, done);
            }
            else
            {
                bar.Report(done, rate);
            }
        };
    }
}
