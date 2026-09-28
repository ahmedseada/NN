namespace Qasd.Api;

/// <summary>The "IntentModel" configuration section.</summary>
public sealed class IntentModelOptions
{
    /// <summary>The model file written by <c>qasd train</c>.</summary>
    public string Path { get; set; } = "models/intents.nsm";

    /// <summary>auto (the GPU when there is one), cpu, cuda or cuda:N.</summary>
    public string Device { get; set; } = "cpu";

    /// <summary>Predictions below this confidence are returned with Accepted = false.</summary>
    public float MinConfidence { get; set; } = 0.6f;

    /// <summary>Most messages per batch request.</summary>
    public int MaxBatch { get; set; } = 256;

    /// <summary>Longest message in characters (longer ones are refused).</summary>
    public int MaxTextLength { get; set; } = 4000;

    /// <summary>The folder written by 'qasd-tuned train' (adapter + qasd-tuned.json); empty: the tuned model is not served.</summary>
    public string? TunedPath { get; set; }

    /// <summary>Where the tuned model runs: cpu (default), cuda, cuda:N or auto (a language model is much faster on the GPU).</summary>
    public string TunedDevice { get; set; } = "cpu";

    /// <summary>Most messages per batch request to the tuned model (it is slower than the classifier).</summary>
    public int MaxTunedBatch { get; set; } = 32;

    /// <summary>The model requests use when they name none: classifier or tuned.</summary>
    public string DefaultModel { get; set; } = "classifier";

    /// <summary>When set, POST /v1/model/reload needs this value in the X-Admin-Key header; when empty, reloading is disabled.</summary>
    public string? AdminKey { get; set; }
}
