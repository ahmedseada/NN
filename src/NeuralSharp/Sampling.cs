namespace NeuralSharp;

/// <summary>One sampled token with the statistics recorded by <see cref="TokenSampler"/>.</summary>
/// <param name="Id">The chosen token id.</param>
/// <param name="Probability">Its probability after temperature and top-k.</param>
/// <param name="Entropy">Entropy of the sampling distribution, in bits.</param>
/// <param name="Alternatives">The five most likely tokens (id, probability), most likely first; id -1 when fewer exist.</param>
public readonly record struct SampledToken(int Id, float Probability, float Entropy, (int Id, float Probability)[] Alternatives);

/// <summary>
/// Samples the next token for a batch of sequences on the device that holds the logits, so the chosen ids can feed
/// the next decoding step without a round trip to the host. Randomness is counter-based (seed, step, row), so
/// results are reproducible and identical on CPU and GPU up to floating-point rounding. Per-token statistics are
/// written to a device buffer and read back in chunks with <see cref="Read"/>.
/// </summary>
public sealed class TokenSampler : IDisposable
{
    private const int StatsPerToken = 13;
    private readonly Tensor _stats;
    private readonly Tensor _step;

    /// <summary>Creates a sampler for <paramref name="rows"/> sequences over <paramref name="vocabulary"/> tokens, for up to <paramref name="maxSteps"/> steps.</summary>
    public TokenSampler(Device device, int rows, int vocabulary, int maxSteps)
    {
        Rows = rows;
        Vocabulary = vocabulary;
        MaxSteps = maxSteps;
        Ids = Tensor.Persistent(new float[rows], [rows], device, requiresGrad: false);
        _stats = Tensor.Persistent(new float[maxSteps * rows * StatsPerToken], [maxSteps, rows, StatsPerToken], device, requiresGrad: false);
        _step = Tensor.Persistent([0f], [1], device, requiresGrad: false);
    }

    /// <summary>Sequences sampled per step.</summary>
    public int Rows { get; }

    /// <summary>Vocabulary size.</summary>
    public int Vocabulary { get; }

    /// <summary>Capacity of the statistics buffer.</summary>
    public int MaxSteps { get; }

    /// <summary>The most recently sampled ids, [rows] (device); reshape to [rows, 1] to feed the next step.</summary>
    public Tensor Ids { get; }

    /// <summary>Softmax temperature (fixed when a step using this sampler is recorded as a graph).</summary>
    public float Temperature { get; set; } = 1f;

    /// <summary>Restrict sampling to the k most likely tokens (0 = all).</summary>
    public int TopK { get; set; }

    /// <summary>Random seed.</summary>
    public uint Seed { get; set; }

    /// <summary>
    /// Samples one token per row from the last position of <paramref name="logits"/> ([rows, vocabulary] or
    /// [rows, steps, vocabulary]), stores ids and statistics, and advances the device-side step counter.
    /// </summary>
    public void Sample(Tensor logits)
    {
        int vocabulary = logits.Shape[^1];
        int steps = logits.Rank == 3 ? logits.Shape[1] : 1;
        if (vocabulary != Vocabulary || logits.Shape[0] != Rows)
        {
            throw new ArgumentException($"Expected [{Rows}, ..., {Vocabulary}] logits, got {Tensor.FormatShape(logits.Shape)}.");
        }

        logits.Backend.SampleRows(logits.Storage, Ids.Storage, _stats.Storage, _step.Storage, Rows, Vocabulary,
            steps * vocabulary, (steps - 1) * vocabulary, Temperature, TopK, Seed);
        _step.AddInPlace(1f);
    }

    /// <summary>Restarts step counting (and the random stream) from step 0.</summary>
    public void Reset() => _step.FillInPlace(0f);

    /// <summary>Downloads the statistics of steps [<paramref name="fromStep"/>, <paramref name="toStep"/>) as [step][row] tokens (one synchronization).</summary>
    public SampledToken[][] Read(int fromStep, int toStep)
    {
        int count = toStep - fromStep;
        var raw = new float[count * Rows * StatsPerToken];
        _stats.CopyTo(raw, fromStep * Rows * StatsPerToken);
        var result = new SampledToken[count][];
        for (int s = 0; s < count; s++)
        {
            result[s] = new SampledToken[Rows];
            for (int r = 0; r < Rows; r++)
            {
                int o = (s * Rows + r) * StatsPerToken;
                var alternatives = new (int, float)[5];
                for (int a = 0; a < 5; a++)
                {
                    alternatives[a] = ((int)raw[o + 3 + 2 * a], raw[o + 4 + 2 * a]);
                }

                result[s][r] = new SampledToken((int)raw[o], raw[o + 1], raw[o + 2], alternatives);
            }
        }

        return result;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Ids.Dispose();
        _stats.Dispose();
        _step.Dispose();
    }
}
