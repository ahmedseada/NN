using Microsoft.Extensions.Options;

namespace Qasd.Api;

/// <summary>The tuned language model (IntentModel:TunedPath), loaded at startup when configured.</summary>
public sealed class TunedHost : IDisposable
{
    public TunedHost(IOptions<IntentModelOptions> options, ILogger<TunedHost> logger)
    {
        string? path = options.Value.TunedPath;
        if (string.IsNullOrWhiteSpace(path))
        {
            Status = "not configured (set IntentModel:TunedPath to a folder from 'qasd-tuned train')";
            return;
        }

        var device = ModelHost.ParseDevice(options.Value.TunedDevice);
        Classifier = TunedClassifier.Load(path, device);
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
