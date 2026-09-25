namespace NeuralSharp;

/// <summary>A snapshot of a device's memory accounting, in bytes.</summary>
/// <param name="InUse">Bytes held by live tensors.</param>
/// <param name="Cached">Bytes of freed blocks kept for reuse (returned to the system by <see cref="ComputeResources.ReleaseCachedMemory"/>).</param>
/// <param name="Limit">The configured cap, or null when unlimited.</param>
public readonly record struct MemoryUsage(long InUse, long Cached, long? Limit)
{
    /// <summary>InUse + Cached: what the library currently holds from the system.</summary>
    public long Reserved => InUse + Cached;

    /// <inheritdoc />
    public override string ToString() =>
        $"in use {Format(InUse)}, cached {Format(Cached)}" + (Limit is { } l ? $", limit {Format(l)}" : "");

    private static string Format(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GiB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):F1} MiB",
        >= 1L << 10 => $"{bytes / (double)(1L << 10):F1} KiB",
        _ => $"{bytes} B",
    };
}

/// <summary>Thrown when an allocation would exceed a limit set on <see cref="ComputeResources"/>.</summary>
public sealed class ResourceLimitExceededException(string message) : Exception(message);

/// <summary>
/// Controls how much of the machine NeuralSharp may use. By default there are no limits:
/// every CPU core is available and memory is allocated as the network needs it.
/// </summary>
/// <example>
/// <code>
/// ComputeResources.MaxCpuThreads = 4;                       // leave the other cores for the app
/// ComputeResources.GpuMemoryLimit = 2L * 1024 * 1024 * 1024; // at most 2 GiB per GPU
/// </code>
/// </example>
public static class ComputeResources
{
    private static ParallelOptions s_parallelOptions = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    /// <summary>
    /// The maximum number of CPU threads used by CPU kernels, data loading and parsing.
    /// Defaults to <see cref="Environment.ProcessorCount"/>; 1 makes all CPU work single-threaded.
    /// </summary>
    public static int MaxCpuThreads
    {
        get => s_parallelOptions.MaxDegreeOfParallelism;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            s_parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Math.Min(value, Environment.ProcessorCount) };
        }
    }

    /// <summary>Maximum bytes of tensor memory on the CPU (null = unlimited).</summary>
    public static long? CpuMemoryLimit { get; set; }

    /// <summary>Maximum bytes of tensor memory on each GPU (null = unlimited, up to what the card has).</summary>
    public static long? GpuMemoryLimit { get; set; }

    internal static ParallelOptions ParallelOptions => s_parallelOptions;

    /// <summary>Whether CPU work should be split across threads at all.</summary>
    internal static bool AllowParallel => s_parallelOptions.MaxDegreeOfParallelism > 1;

    /// <summary>Returns the memory accounting for <paramref name="device"/>.</summary>
    public static MemoryUsage GetMemoryUsage(Device device) => device.Backend.GetMemoryUsage();

    /// <summary>Returns cached (unused) blocks to the system for one device, or every initialized device when null.</summary>
    public static void ReleaseCachedMemory(Device? device = null)
    {
        if (device is not null)
        {
            device.Backend.ReleaseCachedMemory();
            return;
        }

        Device.Cpu.Backend.ReleaseCachedMemory();
        for (int i = 0; i < Device.CudaDeviceCount; i++)
        {
            if (Backends.Cuda.CudaBackend.IsInitialized(i))
            {
                Device.Cuda(i).Backend.ReleaseCachedMemory();
            }
        }
    }
}
