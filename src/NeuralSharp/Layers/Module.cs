using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using NeuralSharp.Diagnostics;

namespace NeuralSharp.Layers;

/// <summary>
/// Base class for network building blocks. A module maps an input tensor to an output tensor
/// and owns the trainable parameters it uses.
/// </summary>
public abstract class Module : IDisposable
{
    private const uint FileMagic = 0x3157_534E; // "NSW1"

    /// <summary>An optional name shown in summaries and telemetry.</summary>
    public string? Name { get; set; }

    /// <summary>The name, or a description of the module when no name was set.</summary>
    public string DisplayName => Name ?? ToString() ?? GetType().Name;

    /// <summary>
    /// True in training mode (the default), false in evaluation mode. Layers such as
    /// <see cref="Dropout"/> behave differently in each. Change it with <see cref="Train"/> and <see cref="Eval"/>.
    /// </summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Computes the module's output for a batch of inputs, recording gradients when autograd is on.
    /// Publishes a <see cref="LayerForward"/> event when <see cref="TelemetryLevel.Layers"/> is enabled.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        if (!Telemetry.IsEnabled(TelemetryLevel.Layers))
        {
            return ForwardCore(input);
        }

        int depth = Telemetry.EnterLayer();
        long start = Stopwatch.GetTimestamp();
        Tensor output;
        try
        {
            output = ForwardCore(input);
        }
        catch
        {
            Telemetry.LeaveLayer(depth);
            throw;
        }

        Telemetry.LayerForward(this, input, output, start, depth);
        return output;
    }

    /// <summary>Implements the forward computation. Override this in custom layers.</summary>
    protected abstract Tensor ForwardCore(Tensor input);

    /// <summary>
    /// Runs inference: evaluation mode, no gradient recording. Publishes an <see cref="InferenceCompleted"/>
    /// event when <see cref="TelemetryLevel.Inference"/> is enabled.
    /// </summary>
    public Tensor Predict(Tensor input)
    {
        bool wasTraining = IsTraining;
        long start = Telemetry.Start(TelemetryLevel.Inference);
        Eval();
        try
        {
            Tensor output;
            using (Autograd.NoGrad())
            using (var scope = new TensorScope())
            {
                // Intermediate activations are released immediately; only the result survives.
                output = Forward(input);
                if (!ReferenceEquals(output, input))
                {
                    scope.Keep(output);
                }
            }

            if (start != 0)
            {
                output.Device.Synchronize();
                Telemetry.Inference(new InferenceCompleted(DisplayName, input.Rank > 0 ? input.Shape[0] : 1,
                    input.Shape.ToArray(), output.Shape.ToArray(), output.Device, Stopwatch.GetElapsedTime(start)));
            }

            return output;
        }
        finally
        {
            Train(wasTraining);
        }
    }

    /// <summary>Predicts a batch given as a [rows, features] array and returns [rows, outputs].</summary>
    public float[,] Predict(float[,] input)
    {
        var device = Parameters().FirstOrDefault()?.Device ?? Device.Default;
        using var x = Tensor.From(input, device);
        using var y = Predict(x);
        return y.ToArray2D();
    }

    /// <summary>Switches this module and its children to training (true) or evaluation (false) mode.</summary>
    public void Train(bool training = true)
    {
        IsTraining = training;
        foreach (var child in Children())
        {
            child.Train(training);
        }
    }

    /// <summary>Switches to evaluation mode (same as <c>Train(false)</c>).</summary>
    public void Eval() => Train(false);

    /// <summary>Direct sub-modules, for containers. Leaf layers return none.</summary>
    public virtual IEnumerable<Module> Children() => [];

    /// <summary>The trainable tensors of this module and its children, in a stable order.</summary>
    public virtual IEnumerable<Tensor> Parameters() => Children().SelectMany(c => c.Parameters());

    /// <summary>A table of layers and parameter counts.</summary>
    public string Summary()
    {
        var sb = new StringBuilder();
        void Walk(Module m, int depth)
        {
            long own = m.Children().Any() ? 0 : m.ParameterCount;
            sb.Append(new string(' ', depth * 2)).Append(m.DisplayName);
            sb.Append(own > 0 ? $"  [{own:N0} params]" : "").AppendLine();
            foreach (var child in m.Children())
            {
                Walk(child, depth + 1);
            }
        }

        Walk(this, 0);
        sb.Append($"Total trainable parameters: {ParameterCount:N0}");
        return sb.ToString();
    }

    /// <summary>The total number of trainable values.</summary>
    public long ParameterCount => Parameters().Sum(p => (long)p.Size);

    /// <summary>Moves every parameter to <paramref name="device"/> and returns this module.</summary>
    public Module To(Device device)
    {
        MoveTo(device);
        return this;
    }

    /// <summary>Moves this module's own parameters; by default forwards the call to the children.</summary>
    protected internal virtual void MoveTo(Device device)
    {
        foreach (var child in Children())
        {
            child.MoveTo(device);
        }
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
            throw new InvalidDataException($"{path} is not a NeuralSharp weights file.");
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
