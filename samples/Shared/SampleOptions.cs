using System.Globalization;
using NeuralSharp;
using NeuralSharp.Diagnostics;

namespace NeuralSharp.Samples;

/// <summary>Command-line options shared by the samples.</summary>
internal sealed record SampleOptions
{
    public Device Device { get; init; } = Device.Default;
    public int? Epochs { get; init; }
    public int? BatchSize { get; init; }
    public TelemetryLevel LogLevels { get; init; } = TelemetryLevel.Training;
    public string? LogFile { get; init; }
    public string? DataFile { get; init; }

    public const string Usage = """
        Options:
          --device <auto|cpu|cuda|cuda:N>  where to run (default: auto = GPU if available, else CPU)
          --cpu / --cuda                   shortcuts for --device cpu / --device cuda
          --threads <n>                    CPU threads to use (default: all cores)
          --gpu-memory <MiB>               cap GPU tensor memory (default: unlimited)
          --cpu-memory <MiB>               cap CPU tensor memory (default: unlimited)
          --epochs <n>                     training epochs
          --batch-size <n>                 samples per batch
          --log <levels>                   comma list: training,batches,gradients,layers,operations,inference,all
          --log-file <path>                also write telemetry as JSON Lines to this file
          --data <path>                    dataset file (house-price sample)
          --help                           show this help
        """;

    /// <summary>Parses the arguments and applies device and resource settings globally. Returns null after printing help.</summary>
    public static SampleOptions? Parse(string[] args)
    {
        try
        {
            return ParseCore(args);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FormatException)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            Environment.Exit(2);
            return null;
        }
    }

    private static SampleOptions? ParseCore(string[] args)
    {
        var options = new SampleOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
            int NextInt() => int.Parse(Next(), CultureInfo.InvariantCulture);

            switch (args[i].ToLowerInvariant())
            {
                case "--help" or "-h" or "-?":
                    Console.WriteLine(Usage);
                    return null;
                case "--cpu":
                    options = options with { Device = Device.Cpu };
                    break;
                case "--cuda" or "--gpu":
                    options = options with { Device = Device.Cuda() };
                    break;
                case "--device":
                    options = options with { Device = ParseDevice(Next()) };
                    break;
                case "--threads":
                    ComputeResources.MaxCpuThreads = NextInt();
                    break;
                case "--gpu-memory":
                    ComputeResources.GpuMemoryLimit = NextInt() * 1024L * 1024;
                    break;
                case "--cpu-memory":
                    ComputeResources.CpuMemoryLimit = NextInt() * 1024L * 1024;
                    break;
                case "--epochs":
                    options = options with { Epochs = NextInt() };
                    break;
                case "--batch-size":
                    options = options with { BatchSize = NextInt() };
                    break;
                case "--log":
                    options = options with { LogLevels = ParseLevels(Next()) };
                    break;
                case "--log-file":
                    options = options with { LogFile = Next() };
                    break;
                case "--data":
                    options = options with { DataFile = Next() };
                    break;
                default:
                    throw new ArgumentException($"Unknown option '{args[i]}'.\n{Usage}");
            }
        }

        Device.Default = options.Device;
        return options;
    }

    private static Device ParseDevice(string text) => text.ToLowerInvariant() switch
    {
        "auto" => Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu,
        "cpu" => Device.Cpu,
        "cuda" or "gpu" => Device.Cuda(),
        var s when s.StartsWith("cuda:", StringComparison.Ordinal) => Device.Cuda(int.Parse(s[5..], CultureInfo.InvariantCulture)),
        _ => throw new ArgumentException($"Unknown device '{text}'. Use auto, cpu, cuda or cuda:N."),
    };

    private static TelemetryLevel ParseLevels(string text) =>
        text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Aggregate(TelemetryLevel.None, (levels, name) => levels | Enum.Parse<TelemetryLevel>(name, ignoreCase: true));
}
