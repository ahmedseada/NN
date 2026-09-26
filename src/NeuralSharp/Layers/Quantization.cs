using System.Runtime.InteropServices;

namespace NeuralSharp.Layers;

/// <summary>How <see cref="Module.Save(string, WeightFormat)"/> stores floating-point values in a weights file.</summary>
public enum WeightFormat
{
    /// <summary>32-bit floats: exact (4 bytes per value).</summary>
    Float32,

    /// <summary>IEEE half precision (2 bytes per value): about 3 significant digits, range ±65504.</summary>
    Float16,

    /// <summary>bfloat16 (2 bytes per value): the range of float32 with about 2–3 significant digits.</summary>
    BFloat16,
}

/// <summary>
/// The int8 weights of a quantized <see cref="Linear"/> layer: each weight is a signed byte times its output column's
/// scale (symmetric, per-column: scale = max |w| / 127). A quarter of the float32 memory; created by
/// <see cref="ModuleExtensions.QuantizeInt8"/>. Inputs, outputs, biases and LoRA adapters stay float32.
/// </summary>
public sealed class Int8Weight : IDisposable
{
    private Int8Weight(Tensor packed, Tensor scales, int rows, int columns)
    {
        Packed = packed;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>Input features (rows of the weight matrix).</summary>
    public int Rows { get; }

    /// <summary>Output features (columns).</summary>
    public int Columns { get; }

    /// <summary>Device memory used, in bytes (weights and scales).</summary>
    public long Bytes => 4L * (Packed.Size + Scales.Size);

    /// <summary>The bytes, packed four per element along each row (rows padded to a multiple of four columns).</summary>
    internal Tensor Packed { get; private set; }

    /// <summary>One scale per column.</summary>
    internal Tensor Scales { get; private set; }

    /// <summary>Quantizes a float weight matrix [rows, columns] (on its device).</summary>
    public static Int8Weight Quantize(Tensor weight)
    {
        if (weight.Rank != 2)
        {
            throw new ArgumentException($"Int8 quantization needs a matrix, got {Tensor.FormatShape(weight.Shape)}.", nameof(weight));
        }

        return Quantize(weight.ToArray(), weight.Shape[0], weight.Shape[1], weight.Device);
    }

    /// <summary>
    /// Quantizes weights given as host values [rows, columns] and uploads only the bytes to <paramref name="device"/>
    /// (large models never exist as float32 on the device).
    /// </summary>
    public static Int8Weight Quantize(ReadOnlySpan<float> values, int rows, int columns, Device device)
    {
        if (values.Length != rows * columns)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{rows}, {columns}].", nameof(values));
        }

        int stride = (columns + 3) / 4 * 4;
        var scales = new float[columns];
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < columns; j++)
            {
                scales[j] = MathF.Max(scales[j], MathF.Abs(values[r * columns + j]));
            }
        }

        for (int j = 0; j < columns; j++)
        {
            scales[j] = scales[j] > 0f ? scales[j] / 127f : 1f;
        }

        var bytes = new sbyte[rows * stride];
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < columns; j++)
            {
                bytes[r * stride + j] = (sbyte)Math.Clamp(MathF.Round(values[r * columns + j] / scales[j]), -127f, 127f);
            }
        }

        var packed = MemoryMarshal.Cast<sbyte, float>(bytes).ToArray();
        return new Int8Weight(
            Tensor.Persistent(packed, [packed.Length], device, requiresGrad: false),
            Tensor.Persistent(scales, [columns], device, requiresGrad: false),
            rows, columns);
    }

    /// <summary>The float weights these bytes stand for, [rows, columns], on the same device.</summary>
    public Tensor Dequantize()
    {
        var w = Tensor.Persistent(new float[Rows * Columns], [Rows, Columns], Packed.Device, requiresGrad: false);
        Packed.Backend.Int8Dequantize(Packed.Storage, Scales.Storage, w.Storage, Rows, Columns);
        return w;
    }

    internal void MoveTo(Device device, Func<Tensor, Device, Tensor> move)
    {
        Packed = move(Packed, device);
        Scales = move(Scales, device);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Packed.Dispose();
        Scales.Dispose();
    }
}
