using Microsoft.Extensions.Options;
using Idrak;

namespace Qasd.Api;

/// <summary>
/// Holds the loaded model and swaps in a new one on reload without disturbing requests in flight: each request leases
/// the current model, and a replaced model is disposed when its last lease ends.
/// </summary>
public sealed class ModelHost : IDisposable
{
    private readonly IntentModelOptions _options;
    private readonly ILogger<ModelHost> _logger;
    private readonly Lock _gate = new();
    private Handle _current;

    public ModelHost(IOptions<IntentModelOptions> options, ILogger<ModelHost> logger)
    {
        _options = options.Value;
        _logger = logger;
        _current = Open(_options.ClassifierPath);
    }

    /// <summary>The model's description.</summary>
    public ModelInfo Info
    {
        get
        {
            using var lease = Lease();
            return new ModelInfo(lease.Path, lease.Classifier.Labels, lease.Classifier.Device.Name, lease.LoadedAt, _options.MinConfidence);
        }
    }

    /// <summary>The current model, kept alive until the lease is disposed.</summary>
    public Handle.Lease Lease()
    {
        lock (_gate)
        {
            return _current.Acquire();
        }
    }

    /// <summary>Loads the model file again (after retraining) and serves it from now on.</summary>
    public ModelInfo Reload()
    {
        var next = Open(_options.ClassifierPath);
        Handle previous;
        lock (_gate)
        {
            previous = _current;
            _current = next;
        }

        previous.Retire();
        return Info;
    }

    public void Dispose() => _current.Retire();

    private Handle Open(string path)
    {
        if (!File.Exists(path))
        {
            string advice = string.IsNullOrWhiteSpace(_options.Path)
                ? "train one with 'qasd train apps/Qasd/data/plan-queries.csv' (it saves there) or set IntentModel:Path."
                : $"IntentModel:Path (appsettings.json, or the IntentModel__Path environment variable) is set to '{_options.Path}'; "
                  + (File.Exists(QasdPaths.Classifier)
                      ? $"remove it to use the trained model at {QasdPaths.Classifier} (PowerShell: Remove-Item Env:IntentModel__Path)."
                      : "point it at a model from 'qasd train', or remove it to use the default place.");
            throw new FileNotFoundException($"No intent model at {Path.GetFullPath(path)}: {advice}", path);
        }

        var device = ParseDevice(_options.Device);
        var classifier = TextClassifier.Load(path, device);
        classifier.Predict(["warm up", "تسخين"]);                                 // compile and allocate now, not on the first request
        _logger.LogInformation("Loaded intent model {Path} on {Device}: {Labels}", Path.GetFullPath(path), device.Name, string.Join(", ", classifier.Labels));
        return new Handle(classifier, Path.GetFullPath(path), DateTimeOffset.UtcNow);
    }

    /// <summary>auto (the GPU when there is one), cpu, cuda or cuda:N.</summary>
    public static Device ParseDevice(string name) => name.ToLowerInvariant() switch
    {
        "auto" => Idrak.Device.IsCudaAvailable ? Idrak.Device.Cuda() : Idrak.Device.Cpu,
        "cpu" => Idrak.Device.Cpu,
        "cuda" or "gpu" => Idrak.Device.Cuda(),
        ['c', 'u', 'd', 'a', ':', .. var index] => Idrak.Device.Cuda(int.Parse(index, System.Globalization.CultureInfo.InvariantCulture)),
        _ => throw new InvalidOperationException($"Device '{name}' is not auto, cpu, cuda or cuda:N."),
    };

    /// <summary>A loaded model with a count of the requests using it.</summary>
    public sealed class Handle(TextClassifier classifier, string path, DateTimeOffset loadedAt)
    {
        private readonly Lock _gate = new();
        private int _leases;
        private bool _retired;

        internal Lease Acquire()
        {
            lock (_gate)
            {
                _leases++;
                return new Lease(this);
            }
        }

        internal void Retire()
        {
            lock (_gate)
            {
                _retired = true;
                if (_leases == 0)
                {
                    classifier.Dispose();
                }
            }
        }

        private void Release()
        {
            lock (_gate)
            {
                if (--_leases == 0 && _retired)
                {
                    classifier.Dispose();
                }
            }
        }

        /// <summary>A request's use of a model.</summary>
        public sealed class Lease(Handle handle) : IDisposable
        {
            private int _released;

            public TextClassifier Classifier => handle.Classifier;

            public string Path => handle.Path;

            public DateTimeOffset LoadedAt => handle.LoadedAt;

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _released, 1) == 0)
                {
                    handle.Release();
                }
            }
        }

        private TextClassifier Classifier => classifier;

        private string Path => path;

        private DateTimeOffset LoadedAt => loadedAt;
    }
}
