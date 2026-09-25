using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace NN.Layers;

/// <summary>
/// Base class for network building blocks. A module maps an input tensor to an output tensor
/// and owns the trainable parameters it uses.
/// </summary>
public abstract class Module : IDisposable
{
    private const uint FileMagic = 0x3157_4E4E; // "NNW1"

    /// <summary>Computes the module's output for a batch of inputs.</summary>
    public abstract Tensor Forward(Tensor input);

    /// <summary>The trainable tensors of this module (and its children), in a stable order.</summary>
    public virtual IEnumerable<Tensor> Parameters() => [];

    /// <summary>The total number of trainable values.</summary>
    public long ParameterCount => Parameters().Sum(p => (long)p.Size);

    /// <summary>Moves every parameter to <paramref name="device"/> and returns this module.</summary>
    public Module To(Device device)
    {
        MoveTo(device);
        return this;
    }

    /// <summary>Moves this module's own parameters; containers forward the call to their children.</summary>
    protected internal virtual void MoveTo(Device device)
    {
    }

    /// <summary>Writes all parameter values to a binary file.</summary>
    public void Save(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        var parameters = Parameters().ToList();
        writer.Write(FileMagic);
        writer.Write(parameters.Count);
        foreach (var p in parameters)
        {
            writer.Write(p.Rank);
            foreach (int d in p.Shape)
            {
                writer.Write(d);
            }

            var data = p.ToArray();
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
        }
    }

    /// <summary>Reads parameter values written by <see cref="Save"/> into this module. The architecture must match.</summary>
    public void Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        var parameters = Parameters().ToList();
        if (reader.ReadUInt32() != FileMagic)
        {
            throw new InvalidDataException($"{path} is not an NN weights file.");
        }

        int count = reader.ReadInt32();
        if (count != parameters.Count)
        {
            throw new InvalidDataException($"The file has {count} parameter tensors but the model has {parameters.Count}.");
        }

        foreach (var p in parameters)
        {
            var shape = new int[reader.ReadInt32()];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            if (!shape.AsSpan().SequenceEqual(p.Shape))
            {
                throw new InvalidDataException($"Shape mismatch: file has {Tensor.FormatShape(shape)}, model has {Tensor.FormatShape(p.Shape)}.");
            }

            var data = new float[p.Size];
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            p.Load(data);
        }
    }

    /// <summary>Releases the device memory held by the parameters.</summary>
    public virtual void Dispose()
    {
        foreach (var p in Parameters())
        {
            p.Dispose();
        }

        GC.SuppressFinalize(this);
    }
}
