using Microsoft.Extensions.Options;

namespace Qasd.Api;

/// <summary>The tuned language model (IntentModel:TunedPath), loaded at startup when configured.</summary>
public sealed class TunedHost : IDisposable
{
    public TunedHost(IOptions<IntentModelOptions> options, ModelHost classifier, ILogger<TunedHost> logger)
    {
        string? path = options.Value.TunedFolder;
        if (path is null)
        {
            Status = "turned off (IntentModel:TunedPath is \"none\")";
            return;
        }

        if (!TunedClassifier.IsTunedFolder(path))
        {
            Status = $"not trained yet: no tuned model in {System.IO.Path.GetFullPath(path)} (tune one with idrak-tune as the README shows, or set IntentModel:TunedPath)";
            logger.LogWarning("Serving the classifier only: {Status}", Status);
            return;
        }

        var device = ModelHost.ParseDevice(options.Value.TunedDevice, logger);
        // The intents are the classifier's: both models answer with the same labels.
        using (var lease = classifier.Lease())
        {
            Classifier = TunedClassifier.Load(path, lease.Classifier.Labels, device);
        }

        Classifier.Predict(["warm up"]);
        LoadedAt = DateTimeOffset.UtcNow;
        Status = Classifier.Folder;
        logger.LogInformation("Loaded tuned model {Folder} (base {Base}) on {Device}: {Labels}", Classifier.Folder, Classifier.BaseModel, device.Name,
            string.Join(", ", Classifier.Labels));
    }

    /// <summary>The tuned model, or null when not configured.</summary>
    public TunedClassifier? Classifier { get; }

    /// <summary>The folder, or why there is no tuned model.</summary>
    public string Status { get; }

    public DateTimeOffset? LoadedAt { get; }

    public void Dispose() => Classifier?.Dispose();
}
