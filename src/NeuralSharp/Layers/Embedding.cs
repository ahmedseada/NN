namespace NeuralSharp.Layers;

/// <summary>
/// Maps integer token ids to learned vectors: [..., ] ids → [..., dim]. Ids are stored as floats
/// (exact up to 16,777,216) and must lie in [0, vocabulary). On the CPU an invalid id throws; on the GPU
/// it is clamped to the valid range, so validate ids when loading data.
/// </summary>
public sealed class Embedding : Module
{
    /// <summary>Creates a table of <paramref name="vocabulary"/> vectors of size <paramref name="dim"/>, initialized N(0, 1).</summary>
    public Embedding(int vocabulary, int dim, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabulary);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        Vocabulary = vocabulary;
        Dim = dim;
        random ??= Random.Shared;
        var values = new float[vocabulary * dim];
        for (int i = 0; i < values.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }

        Weight = CreateParameter(values, [vocabulary, dim], device ?? Device.Default);
    }

    /// <summary>Number of distinct ids.</summary>
    public int Vocabulary { get; }

    /// <summary>Vector size.</summary>
    public int Dim { get; }

    /// <summary>The [vocabulary, dim] table.</summary>
    public Tensor Weight { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Weight.EmbeddingLookup(input);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [Weight];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device) => Weight = MoveTensor(Weight, device);

    /// <inheritdoc />
    public override string ToString() => $"Embedding({Vocabulary} -> {Dim})";
}
