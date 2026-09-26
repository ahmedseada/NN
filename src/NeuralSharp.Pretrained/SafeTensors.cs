using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NeuralSharp.Pretrained;

/// <summary>The element type of a stored tensor.</summary>
public enum SafeTensorType
{
    /// <summary>32-bit float.</summary>
    F32,

    /// <summary>16-bit IEEE float.</summary>
    F16,

    /// <summary>bfloat16.</summary>
    BF16,
}

/// <summary>A tensor's entry in a safetensors file: its type, shape and byte range.</summary>
public sealed record SafeTensorInfo(string Name, SafeTensorType Type, int[] Shape, long Offset, long Length, string File)
{
    /// <summary>Number of elements.</summary>
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reads safetensors weights: one file, or a sharded checkpoint (<c>model.safetensors.index.json</c> naming the file of
/// each tensor). Tensors are read one at a time, straight from disk, and converted to float32.
/// </summary>
public sealed class SafeTensorsReader : IDisposable
{
    private readonly Dictionary<string, SafeTensorInfo> _tensors = [];
    private readonly Dictionary<string, FileStream> _files = [];

    private SafeTensorsReader()
    {
    }

    /// <summary>The tensors, by name.</summary>
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors => _tensors;

    /// <summary>Metadata of the files (the "__metadata__" entries), merged.</summary>
    public Dictionary<string, string> Metadata { get; } = [];

    /// <summary>
    /// Opens a .safetensors file, a sharded checkpoint's index (.index.json), or a folder containing either
    /// (model.safetensors or model.safetensors.index.json).
    /// </summary>
    public static SafeTensorsReader Open(string path)
    {
        if (Directory.Exists(path))
        {
            string index = Path.Combine(path, "model.safetensors.index.json"), single = Path.Combine(path, "model.safetensors");
            path = File.Exists(index) ? index : File.Exists(single) ? single
                : Directory.GetFiles(path, "*.safetensors") is [var only] ? only
                : throw new FileNotFoundException($"{path} has no model.safetensors, model.safetensors.index.json or single .safetensors file.");
        }

        var reader = new SafeTensorsReader();
        try
        {
            if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                var map = JsonNode.Parse(File.ReadAllText(path))!["weight_map"]!.AsObject();
                string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
                foreach (var file in map.Select(p => (string)p.Value!).Distinct())
                {
                    reader.ReadHeader(Path.Combine(folder, file));
                }
            }
            else
            {
                reader.ReadHeader(path);
            }

            return reader;
        }
        catch
        {
            reader.Dispose();
            throw;
        }
    }

    /// <summary>Whether a tensor named <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>The tensor <paramref name="name"/> as float32 values (row-major, as stored).</summary>
    public float[] Read(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"No tensor named '{name}'.");
        var bytes = new byte[info.Length];
        var stream = _files[info.File];
        lock (stream)
        {
            stream.Position = info.Offset;
            stream.ReadExactly(bytes);
        }

        return Decode(bytes, info.Type, checked((int)info.Count));
    }

    private void ReadHeader(string file)
    {
        var stream = File.OpenRead(file);
        _files[file] = stream;
        Span<byte> size = stackalloc byte[8];
        stream.ReadExactly(size);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(size);
        if (headerLength <= 0 || headerLength > 100_000_000)
        {
            throw new InvalidDataException($"{file} is not a safetensors file (header size {headerLength}).");
        }

        var header = new byte[headerLength];
        stream.ReadExactly(header);
        long dataStart = 8 + headerLength;
        foreach (var (name, node) in JsonNode.Parse(header)!.AsObject())
        {
            if (name == "__metadata__")
            {
                foreach (var (key, value) in node!.AsObject())
                {
                    Metadata[key] = value?.ToString() ?? "";
                }

                continue;
            }

            string dtype = (string)node!["dtype"]!;
            var type = dtype switch
            {
                "F32" => SafeTensorType.F32,
                "F16" => SafeTensorType.F16,
                "BF16" => SafeTensorType.BF16,
                _ => throw new NotSupportedException($"Tensor '{name}' in {file} has type {dtype}; F32, F16 and BF16 are supported."),
            };
            int[] shape = [.. node["shape"]!.AsArray().Select(d => (int)d!)];
            var offsets = node["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            _tensors[name] = new SafeTensorInfo(name, type, shape, dataStart + begin, end - begin, file);
        }
    }

    internal static float[] Decode(byte[] bytes, SafeTensorType type, int count)
    {
        var values = new float[count];
        switch (type)
        {
            case SafeTensorType.F32:
                for (int i = 0; i < count; i++)
                {
                    values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
                }

                break;
            case SafeTensorType.F16:
                for (int i = 0; i < count; i++)
                {
                    values[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.AsSpan(i * 2));
                }

                break;
            default:
                for (int i = 0; i < count; i++)
                {
                    values[i] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2)) << 16);
                }

                break;
        }

        return values;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var stream in _files.Values)
        {
            stream.Dispose();
        }

        _files.Clear();
    }
}

/// <summary>Writes safetensors files (for example fine-tuned weights, for other tools).</summary>
public static class SafeTensorsWriter
{
    /// <summary>Writes <paramref name="tensors"/> (name, shape, values) to <paramref name="path"/> in <paramref name="type"/>.</summary>
    public static void Write(string path, IEnumerable<(string Name, int[] Shape, float[] Values)> tensors, SafeTensorType type = SafeTensorType.BF16,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var list = tensors.ToList();
        int size = type == SafeTensorType.F32 ? 4 : 2;
        var header = new JsonObject();
        if (metadata is not null)
        {
            var meta = new JsonObject();
            foreach (var (key, value) in metadata)
            {
                meta[key] = value;
            }

            header["__metadata__"] = meta;
        }

        long offset = 0;
        foreach (var (name, shape, values) in list)
        {
            if (values.Length != shape.Aggregate(1, (a, b) => a * b))
            {
                throw new ArgumentException($"'{name}': {values.Length} values do not fill [{string.Join(", ", shape)}].");
            }

            header[name] = new JsonObject
            {
                ["dtype"] = type.ToString(),
                ["shape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
                ["data_offsets"] = new JsonArray(offset, offset + (long)values.Length * size),
            };
            offset += (long)values.Length * size;
        }

        var headerBytes = Encoding.UTF8.GetBytes(header.ToJsonString());
        int padding = (8 - headerBytes.Length % 8) % 8;                      // the data starts 8-byte aligned
        using var stream = File.Create(path);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(length, headerBytes.Length + padding);
        stream.Write(length);
        stream.Write(headerBytes);
        stream.Write(Encoding.ASCII.GetBytes(new string(' ', padding)));
        foreach (var (_, _, values) in list)
        {
            var bytes = new byte[values.Length * size];
            for (int i = 0; i < values.Length; i++)
            {
                switch (type)
                {
                    case SafeTensorType.F32:
                        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
                        break;
                    case SafeTensorType.F16:
                        BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(i * 2), (Half)values[i]);
                        break;
                    default:
                        uint bits = BitConverter.SingleToUInt32Bits(values[i]);
                        ushort b16 = float.IsNaN(values[i]) ? (ushort)0x7FC0 : (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
                        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), b16);
                        break;
                }
            }

            stream.Write(bytes);
        }
    }
}
