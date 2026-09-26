using System.Text.Json.Nodes;
using NeuralSharp.Inference;
using NeuralSharp.Layers;

namespace NeuralSharp.Onnx;

/// <summary>
/// An ONNX model rebuilt from NeuralSharp layers: the <see cref="Network"/> description, the <see cref="Model"/> with
/// the file's weights (on the device chosen at import, so it runs on NeuralSharp's own CPU or CUDA kernels), and the
/// file's metadata. Dispose it to release the model.
/// </summary>
public sealed class ImportedNetwork : IDisposable
{
    internal ImportedNetwork(NetworkBuilder network, Sequential model, IReadOnlyDictionary<string, string> metadata, IReadOnlyList<string> notes)
    {
        Network = network;
        Model = model;
        Metadata = metadata;
        Notes = notes;
    }

    /// <summary>The layers as builder steps (can be written to JSON and rebuilt).</summary>
    public NetworkBuilder Network { get; }

    /// <summary>The network with the file's weights.</summary>
    public Sequential Model { get; }

    /// <summary>The ONNX file's metadata (custom key/value pairs).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Approximations made during import (for example exact GELU replaced by the tanh approximation).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>
    /// Saves a NeuralSharp package (.nsm): the architecture, the weights and the ONNX metadata (as the JSON entry
    /// "onnx-metadata"). Load it with <c>ModelPackage.Open(path).BuildNetwork(device)</c> or <c>Predictor.Load(path, device)</c>.
    /// </summary>
    public void SavePackage(string path)
    {
        var metadata = new JsonObject();
        foreach (var (key, value) in Metadata)
        {
            metadata[key] = value;
        }

        ModelPackage.Create(path).Architecture(Network).Weights(ModelPackage.DefaultModelName, Model).Json("onnx-metadata", metadata).Save();
    }

    /// <inheritdoc />
    public void Dispose() => Model.Dispose();
}

/// <summary>
/// Imports ONNX models into NeuralSharp layers. The graph must be a chain of supported layers from one input to one
/// output: MatMul/Gemm (+ Add) → Linear; Relu, Tanh, Sigmoid, Softmax; GELU (the Gelu op, the tanh form, or the erf
/// form, which becomes the tanh approximation); BatchNormalization; LayerNormalization; Conv (square kernel, one
/// group); MaxPool; GlobalAveragePool (+ Flatten); Flatten; Gather on a weight table → Embedding; sinusoidal
/// position tables → PositionalEncoding; LSTM and GRU (forward, PyTorch-style GRU); ReduceMean over time;
/// first/last time step; Reshape; Dropout and Identity (skipped); and the attention and transformer-layer blocks
/// NeuralSharp exports. Anything else is reported with the node that could not be imported.
/// </summary>
public static class OnnxImport
{
    /// <summary>Imports an .onnx file onto <paramref name="device"/> (the default device when null).</summary>
    /// <param name="path">The .onnx file.</param>
    /// <param name="device">Where the layers are created.</param>
    /// <param name="sampleShape">The shape of one input sample, when the file leaves dimensions other than the batch dynamic.</param>
    public static ImportedNetwork Load(string path, Device? device = null, int[]? sampleShape = null) =>
        Load(File.ReadAllBytes(path), device, sampleShape);

    /// <summary>Imports a model from the bytes of an .onnx file.</summary>
    public static ImportedNetwork Load(byte[] model, Device? device = null, int[]? sampleShape = null) =>
        new Importer(OnnxModel.Read(model), device ?? Device.Default, sampleShape).Run();
}

internal sealed class Importer(OnnxModel model, Device device, int[]? sampleShape)
{
    private static readonly int[] LstmOrder = [0, 3, 1, 2];   // ONNX gate g holds our gate LstmOrder[g] (ONNX i, o, f, c; ours i, f, c, o)
    private static readonly int[] GruOrder = [1, 0, 2];       // ONNX z, r, h; ours r, z, h

    private readonly Dictionary<string, List<OnnxNode>> _users = model.Nodes
        .SelectMany(n => n.Inputs.Where(i => i.Length > 0).Distinct().Select(i => (i, n))).GroupBy(p => p.i).ToDictionary(g => g.Key, g => g.Select(p => p.n).ToList());

    private readonly HashSet<OnnxNode> _consumed = [];
    private readonly List<Action<Module>?> _loaders = [];
    private readonly List<string> _notes = [];
    private NetworkBuilder _network = null!;

    public ImportedNetwork Run()
    {
        if (model.Inputs.Count != 1 || model.Outputs.Count != 1)
        {
            throw new NotSupportedException($"The importer needs one input and one output; the model has {model.Inputs.Count} and {model.Outputs.Count}.");
        }

        var input = model.Inputs[0];
        int[] shape = sampleShape ?? (input.Shape is { Length: > 1 } s ? s[1..] : throw new NotSupportedException("The input's shape is unknown; pass sampleShape."));
        if (shape.Any(d => d <= 0))
        {
            throw new NotSupportedException($"The input's sample shape [{string.Join(", ", shape)}] has dynamic dimensions; pass sampleShape.");
        }

        bool tokens = Users(input.Name).Any(n => n.Op == "Cast" || (n.Op == "Gather" && n.Inputs[1] == input.Name && Const(n.Inputs[0]) is not null));
        _network = (tokens, shape.Length) switch
        {
            (true, 1) => Layers.Network.Tokens(shape[0]),
            (false, 1) => Layers.Network.Input(shape[0]),
            (false, 2) => Layers.Network.Sequence(shape[0], shape[1]),
            (false, 3) => Layers.Network.Image(shape[0], shape[1], shape[2]),
            _ => throw new NotSupportedException($"Unsupported input shape [{string.Join(", ", shape)}]."),
        };
        _network.OnDevice(device).Seed(0).Named(string.IsNullOrEmpty(model.GraphName) ? "onnx" : model.GraphName);

        string current = input.Name, output = model.Outputs[0].Name;
        while (current != output)
        {
            current = Step(current);
        }

        var built = _network.Build();
        try
        {
            var modules = built.ToList();
            if (modules.Count != _loaders.Count)
            {
                throw new InvalidOperationException($"Internal error: {modules.Count} layers for {_loaders.Count} imported steps.");
            }

            using (Autograd.NoGrad())
            {
                for (int i = 0; i < modules.Count; i++)
                {
                    _loaders[i]?.Invoke(modules[i]);
                }
            }
        }
        catch
        {
            built.Dispose();
            throw;
        }

        return new ImportedNetwork(_network, built, model.Metadata, _notes);
    }

    // ------------------------------------------------------------------ the chain

    private string Step(string x)
    {
        var users = Users(x);
        if (users.Count == 0)
        {
            throw new NotSupportedException($"The value '{x}' is not used by any node and is not the graph output.");
        }

        if (TryTransformer(x) is { } t)
        {
            return t;
        }

        if (ParseAttention(x) is { } attention)
        {
            Consume(attention.Nodes);
            return Push(b => b.MultiHeadAttention(attention.Heads, attention.Causal), m => LoadAttention(m, attention), attention.Output);
        }

        if (TryRecurrent(x) is { } r)
        {
            return r;
        }

        if (ParseGelu(x) is { } gelu)
        {
            Consume(gelu.Nodes);
            return Push(b => b.GELU(), null, gelu.Output);
        }

        if (ParseLinear(x) is { } linear)
        {
            Consume(linear.Nodes);
            return Push(b => b.Linear(linear.Out, bias: linear.Bias is not null), m => LoadLinear((Linear)m, linear), linear.Output);
        }

        if (users.Count != 1)
        {
            throw new NotSupportedException($"'{x}' feeds {users.Count} nodes ({string.Join(", ", users)}); only chains of layers can be imported.");
        }

        var node = users[0];
        string Out() => node.Outputs[0];
        OnnxTensor? Input(int i) => i < node.Inputs.Count && node.Inputs[i].Length > 0 ? Const(node.Inputs[i]) : null;
        Consume(node);
        switch (node.Op)
        {
            case "Identity" or "Dropout":
                return Out();
            case "Relu":
                return Push(b => b.ReLU(), null, Out());
            case "Tanh":
                return Push(b => b.Tanh(), null, Out());
            case "Sigmoid":
                return Push(b => b.Sigmoid(), null, Out());
            case "Softmax":
                long axis = node.Int("axis", model.Opset >= 13 ? -1 : 1);
                if (axis != -1 && axis != _network.CurrentShape.Count)
                {
                    throw Unsupported(node, $"softmax over axis {axis} (only the last axis is supported)");
                }

                return Push(b => b.Softmax(), null, Out());
            case "BatchNormalization":
                var (gamma, beta, mean, variance) = (Input(1), Input(2), Input(3), Input(4));
                if (gamma is null || beta is null || mean is null || variance is null)
                {
                    throw Unsupported(node, "batch normalization with computed statistics");
                }

                float momentum = 1f - node.Float("momentum", 0.9f);
                return Push(b => b.BatchNorm(momentum, node.Float("epsilon", 1e-5f)), m =>
                {
                    var bn = (BatchNorm)m;
                    bn.Gamma.Load(gamma.AsFloats());
                    bn.Beta.Load(beta.AsFloats());
                    bn.RunningMean.Load(mean.AsFloats());
                    bn.RunningVariance.Load(variance.AsFloats());
                }, Out());
            case "LayerNormalization":
                if (node.Int("axis", -1) is not (-1) && node.Int("axis", -1) != _network.CurrentShape.Count)
                {
                    throw Unsupported(node, "layer normalization over more than the last axis");
                }

                var (scale, shift) = (Input(1) ?? throw Unsupported(node, "computed scale"), Input(2));
                return Push(b => b.LayerNorm(node.Float("epsilon", 1e-5f)), m =>
                {
                    var ln = (LayerNorm)m;
                    ln.Gamma.Load(scale.AsFloats());
                    ln.Beta.Load(shift?.AsFloats() ?? new float[ln.Features]);
                }, Out());
            case "Conv":
                return Conv(node, Input(1) ?? throw Unsupported(node, "computed weights"), Input(2));
            case "MaxPool":
                var (k, stride, padding) = Window(node);
                if (node.Int("ceil_mode", 0) != 0 || node.Outputs.Count > 1 && node.Outputs[1].Length > 0 && Users(node.Outputs[1]).Count > 0)
                {
                    throw Unsupported(node, "ceil_mode or indices output");
                }

                return Push(b => b.MaxPool2d(k, stride, padding), null, Out());
            case "GlobalAveragePool":
                var next = Users(Out()) is [var flatten] && (flatten.Op == "Flatten" && flatten.Int("axis", 1) == 1
                    || flatten.Op == "Reshape" && Const(flatten.Inputs[1]) is { } target && target.AsLongs() is [-1 or 0, _]
                    || flatten.Op == "Squeeze")
                    ? flatten
                    : throw Unsupported(node, "global average pooling that is not followed by a flatten");
                Consume(next);
                return Push(b => b.GlobalAveragePool2d(), null, next.Outputs[0]);
            case "Flatten":
                return node.Int("axis", 1) == 1 ? Push(b => b.Flatten(), null, Out()) : throw Unsupported(node, "flatten from an axis other than 1");
            case "Cast":
                if (Users(Out()) is [{ Op: "Gather" } gather] && Const(gather.Inputs[0]) is { Dims.Length: 2 } table && gather.Inputs[1] == Out())
                {
                    Consume(gather);
                    return Push(b => b.Embedding(table.Dims[0], table.Dims[1]), m => ((Embedding)m).Weight.Load(table.AsFloats()), gather.Outputs[0]);
                }

                throw Unsupported(node, "a cast that does not feed an embedding lookup");
            case "Gather":
                return Gather(node, x);
            case "Add":
                return PositionTable(node, x);
            case "ReduceMean":
                var axes = node.Ints("axes") ?? (node.Inputs.Count > 1 ? Const(node.Inputs[1])?.AsLongs() : null);
                if (axes is [1] && node.Int("keepdims", 1) == 0)
                {
                    return Push(b => b.MeanOverTime(), null, Out());
                }

                throw Unsupported(node, "a mean over axes other than time");
            case "Reshape":
                var shape = Input(1)?.AsLongs() ?? throw Unsupported(node, "a computed shape");
                if (shape.Length < 2 || shape[0] is not (-1 or 0) || shape[1..].Any(d => d <= 0))
                {
                    throw Unsupported(node, $"reshape to [{string.Join(", ", shape)}] (the batch must stay first, other sizes fixed)");
                }

                return Push(b => b.Reshape([.. shape[1..].Select(d => (int)d)]), null, Out());
            default:
                throw Unsupported(node, $"the {node.Op} operator");
        }
    }

    private string Push(Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load, string output)
    {
        step(_network);
        _loaders.Add(load);
        return output;
    }

    private List<OnnxNode> Users(string value) => _users.TryGetValue(value, out var users) ? [.. users.Where(u => !_consumed.Contains(u))] : [];

    private OnnxNode? Only(string value, string op) => Users(value) is [var node] && node.Op == op ? node : null;

    private OnnxTensor? Const(string name) => name.Length > 0 && model.Constants.TryGetValue(name, out var t) ? t : null;

    private float? Scalar(string name) => Const(name) is { Size: 1 } t ? t.AsFloats()[0] : null;

    private void Consume(params IEnumerable<OnnxNode> nodes)
    {
        foreach (var node in nodes)
        {
            _consumed.Add(node);
        }
    }

    private static NotSupportedException Unsupported(OnnxNode node, string what) =>
        new($"Cannot import {node}: {what} has no NeuralSharp layer.");

    // The other input of a binary node whose one input is `x`.
    private static string? Other(OnnxNode node, string x) =>
        node.Inputs.Count == 2 && node.Inputs[0] == x ? node.Inputs[1] : node.Inputs.Count == 2 && node.Inputs[1] == x ? node.Inputs[0] : null;

    // ------------------------------------------------------------------ Linear: MatMul (+ Add) or Gemm

    private sealed record LinearMatch(OnnxNode[] Nodes, float[] Weight, float[]? Bias, int In, int Out, string Output);

    private LinearMatch? ParseLinear(string x)
    {
        foreach (var node in Users(x))
        {
            if (node.Op == "MatMul" && node.Inputs[0] == x && Const(node.Inputs[1]) is { Dims.Length: 2 } w)
            {
                var (inF, outF) = (w.Dims[0], w.Dims[1]);
                if (Users(node.Outputs[0]) is [{ Op: "Add" } add] && Other(add, node.Outputs[0]) is { } b && Const(b) is { } bias && bias.Size == outF)
                {
                    return new([node, add], w.AsFloats(), bias.AsFloats(), inF, outF, add.Outputs[0]);
                }

                return new([node], w.AsFloats(), null, inF, outF, node.Outputs[0]);
            }

            if (node.Op == "Gemm" && node.Inputs[0] == x && Const(node.Inputs[1]) is { Dims.Length: 2 } g
                && node.Int("transA", 0) == 0 && node.Float("alpha", 1f) == 1f && node.Float("beta", 1f) == 1f)
            {
                bool transposed = node.Int("transB", 0) == 1;
                var (inF, outF) = transposed ? (g.Dims[1], g.Dims[0]) : (g.Dims[0], g.Dims[1]);
                var values = g.AsFloats();
                var weight = transposed ? Transpose(values, outF, inF) : values;
                var bias = node.Inputs.Count > 2 ? Const(node.Inputs[2]) : null;
                if (bias is not null && bias.Size != outF)
                {
                    continue;
                }

                return new([node], weight, bias?.AsFloats(), inF, outF, node.Outputs[0]);
            }
        }

        return null;
    }

    private static float[] Transpose(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }

    private static void LoadLinear(Linear linear, LinearMatch match)
    {
        linear.Weight.Load(match.Weight);
        if (match.Bias is { } bias)
        {
            linear.Bias!.Load(bias);
        }
    }

    // ------------------------------------------------------------------ GELU

    private sealed record GeluMatch(OnnxNode[] Nodes, string Output);

    private GeluMatch? ParseGelu(string x)
    {
        var users = Users(x);
        if (users is [{ Op: "Gelu" } op])
        {
            if (op.String("approximate") != "tanh")
            {
                _notes.Add($"{op}: exact GELU imported as the tanh approximation (differences below 0.001).");
            }

            return new([op], op.Outputs[0]);
        }

        // Tanh form (NeuralSharp's export): 0.5 · x · (1 + tanh(k (x + c x³))).
        if (users.FirstOrDefault(n => n.Op == "Mul" && n.Inputs.Count == 2 && n.Inputs[0] == x && n.Inputs[1] == x) is { } square
            && Only(square.Outputs[0], "Mul") is { } cube
            && Only(cube.Outputs[0], "Mul") is { } scaled && Near(Scalar(Other(scaled, cube.Outputs[0]) ?? ""), 0.044715f)
            && Only(scaled.Outputs[0], "Add") is { } inner && Other(inner, scaled.Outputs[0]) == x
            && Only(inner.Outputs[0], "Mul") is { } k && Near(Scalar(Other(k, inner.Outputs[0]) ?? ""), 0.7978845608f)
            && Only(k.Outputs[0], "Tanh") is { } tanh
            && Only(tanh.Outputs[0], "Add") is { } plusOne && Near(Scalar(Other(plusOne, tanh.Outputs[0]) ?? ""), 1f)
            && Only(plusOne.Outputs[0], "Mul") is { } gate && Other(gate, plusOne.Outputs[0]) == x
            && Only(gate.Outputs[0], "Mul") is { } half && Near(Scalar(Other(half, gate.Outputs[0]) ?? ""), 0.5f))
        {
            return new([square, cube, scaled, inner, k, tanh, plusOne, gate, half], half.Outputs[0]);
        }

        // Erf form (PyTorch): x · 0.5 · (1 + erf(x / √2)), with the 0.5 applied first or last.
        if (users.FirstOrDefault(n => n.Op == "Div" && n.Inputs[0] == x && Near(Scalar(n.Inputs[1]), MathF.Sqrt(2f))) is { } div
            && Only(div.Outputs[0], "Erf") is { } erf
            && Only(erf.Outputs[0], "Add") is { } add && Near(Scalar(Other(add, erf.Outputs[0]) ?? ""), 1f)
            && Only(add.Outputs[0], "Mul") is { } product)
        {
            string? factor = Other(product, add.Outputs[0]);
            var nodes = new List<OnnxNode> { div, erf, add, product };
            string output = product.Outputs[0];
            bool halfFirst = factor is not null && users.Any(n => n.Op == "Mul" && n.Outputs[0] == factor && Near(Scalar(Other(n, x) ?? ""), 0.5f));
            if (halfFirst)
            {
                nodes.Add(users.First(n => n.Outputs[0] == factor));
            }
            else if (factor == x && Only(output, "Mul") is { } last && Near(Scalar(Other(last, output) ?? ""), 0.5f))
            {
                nodes.Add(last);
                output = last.Outputs[0];
            }
            else
            {
                return null;
            }

            _notes.Add($"{div}: exact (erf) GELU imported as the tanh approximation (differences below 0.001).");
            return new([.. nodes], output);
        }

        return null;
    }

    private static bool Near(float? value, float expected) => value is { } v && MathF.Abs(v - expected) <= 1e-4f * MathF.Max(1f, MathF.Abs(expected));

    // ------------------------------------------------------------------ attention and transformer layers (NeuralSharp's export)

    private sealed record AttentionMatch(OnnxNode[] Nodes, int Heads, bool Causal, LinearMatch Qkv, LinearMatch Projection, string Output);

    private AttentionMatch? ParseAttention(string x)
    {
        if (ParseLinear(x) is not { } qkv || qkv.Out != 3 * qkv.In
            || Only(qkv.Output, "Reshape") is not { } split || Const(split.Inputs[1])?.AsLongs() is not [_, var t, 3, var h, var dh]
            || Only(split.Outputs[0], "Transpose") is not { } heads
            || Users(heads.Outputs[0]) is not { Count: 3 } parts || parts.Any(p => p.Op != "Gather"))
        {
            return null;
        }

        OnnxNode? Part(long i) => parts.FirstOrDefault(p => Const(p.Inputs[1])?.AsLongs() is [var v] && v == i);
        if (Part(0) is not { } q || Part(1) is not { } k || Part(2) is not { } v
            || Only(k.Outputs[0], "Transpose") is not { } kT
            || Only(q.Outputs[0], "MatMul") is not { } scores || scores.Inputs[1] != kT.Outputs[0]
            || Only(scores.Outputs[0], "Mul") is not { } scale || !Near(Scalar(Other(scale, scores.Outputs[0]) ?? ""), 1f / MathF.Sqrt(dh)))
        {
            return null;
        }

        var nodes = new List<OnnxNode>(qkv.Nodes) { split, heads, q, k, v, kT, scores, scale };
        string logits = scale.Outputs[0];
        bool causal = false;
        if (Only(logits, "Add") is { } mask && Const(Other(mask, logits) ?? "") is { Dims: [var rows, var columns] } && rows == t && columns == t)
        {
            nodes.Add(mask);
            logits = mask.Outputs[0];
            causal = true;
        }

        if (Only(logits, "Softmax") is not { } softmax
            || Only(softmax.Outputs[0], "MatMul") is not { } context || context.Inputs[1] != v.Outputs[0]
            || Only(context.Outputs[0], "Transpose") is not { } merge
            || Only(merge.Outputs[0], "Reshape") is not { } join
            || ParseLinear(join.Outputs[0]) is not { } projection)
        {
            return null;
        }

        nodes.AddRange([softmax, context, merge, join, .. projection.Nodes]);
        return new([.. nodes], (int)h, causal, qkv, projection, projection.Output);
    }

    private static void LoadAttention(Module module, AttentionMatch match)
    {
        var children = module.Children().ToList();
        LoadLinear((Linear)children[0], match.Qkv);
        LoadLinear((Linear)children[1], match.Projection);
    }

    private string? TryTransformer(string x)
    {
        var users = Users(x);
        if (users.Count != 2 || users.FirstOrDefault(n => n.Op == "LayerNormalization" && n.Inputs[0] == x) is not { } norm1
            || users.FirstOrDefault(n => n.Op == "Add") is not { } residual1
            || ParseAttention(norm1.Outputs[0]) is not { } attention || Other(residual1, x) != attention.Output)
        {
            return null;
        }

        string a = residual1.Outputs[0];
        var after = Users(a);
        if (after.Count != 2 || after.FirstOrDefault(n => n.Op == "LayerNormalization" && n.Inputs[0] == a) is not { } norm2
            || after.FirstOrDefault(n => n.Op == "Add") is not { } residual2
            || ParseLinear(norm2.Outputs[0]) is not { } ff1)
        {
            return null;
        }

        // Claim the first part so the GELU and second projection are matched on what remains.
        Consume([norm1, residual1, norm2, .. attention.Nodes, .. ff1.Nodes]);
        if (ParseGelu(ff1.Output) is not { } gelu || ParseLinear(gelu.Output) is not { } ff2 || Other(residual2, a) != ff2.Output)
        {
            throw new NotSupportedException($"Cannot import the transformer layer at {norm1}: its feed-forward block is not Linear → GELU → Linear.");
        }

        Consume([residual2, .. gelu.Nodes, .. ff2.Nodes]);
        var (first, second) = (norm1, norm2);
        return Push(b => b.TransformerEncoderLayer(attention.Heads, ff1.Out, dropout: 0f, attention.Causal), m =>
        {
            var c = m.Children().ToList();
            LoadNorm((LayerNorm)c[0], first);
            LoadAttention(c[1], attention);
            LoadNorm((LayerNorm)c[2], second);
            LoadLinear((Linear)c[3], ff1);
            LoadLinear((Linear)c[4], ff2);
        }, residual2.Outputs[0]);
    }

    private void LoadNorm(LayerNorm norm, OnnxNode node)
    {
        norm.Gamma.Load(Const(node.Inputs[1])!.AsFloats());
        norm.Beta.Load(node.Inputs.Count > 2 && Const(node.Inputs[2]) is { } beta ? beta.AsFloats() : new float[norm.Features]);
    }

    // ------------------------------------------------------------------ recurrent layers

    private string? TryRecurrent(string x)
    {
        if (Only(x, "Transpose") is not { } timeMajor || timeMajor.Ints("perm") is not [1, 0, 2]
            || Users(timeMajor.Outputs[0]) is not [{ Op: "LSTM" or "GRU" } rnn] || rnn.Inputs[0] != timeMajor.Outputs[0])
        {
            return null;
        }

        bool lstm = rnn.Op == "LSTM";
        if (rnn.String("direction") is { } direction && direction != "forward" || rnn.Int("layout", 0) != 0 || rnn.Attributes.ContainsKey("activations")
            || rnn.Attributes.ContainsKey("clip") || rnn.Inputs.Skip(4).Any(i => i.Length > 0 && Const(i) is not { } c || i.Length > 0 && Const(i)!.AsFloats().Any(v => v != 0f))
            || !lstm && rnn.Int("linear_before_reset", 0) != 1 || lstm && rnn.Int("input_forget", 0) != 0)
        {
            throw Unsupported(rnn, "this recurrent configuration (only forward, default activations, no peepholes, zero initial state and, for GRU, linear_before_reset = 1)");
        }

        int h = (int)rnn.Int("hidden_size", 0), gates = lstm ? 4 : 3;
        var order = lstm ? LstmOrder : GruOrder;
        var w = Const(rnn.Inputs[1]) ?? throw Unsupported(rnn, "computed weights");
        var rw = Const(rnn.Inputs[2]) ?? throw Unsupported(rnn, "computed weights");
        var b = rnn.Inputs.Count > 3 ? Const(rnn.Inputs[3]) : null;
        int inputs = w.Dims[2];
        float[] Regroup(float[] source, int rows)
        {
            var result = new float[rows * gates * h];                                  // ours: [rows, G·H]
            for (int gate = 0; gate < gates; gate++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (int r = 0; r < rows; r++)
                    {
                        result[r * gates * h + order[gate] * h + j] = source[(gate * h + j) * rows + r];
                    }
                }
            }

            return result;
        }

        var bias = new float[gates * h];
        if (b is not null)
        {
            var values = b.AsFloats();
            for (int gate = 0; gate < gates; gate++)
            {
                for (int j = 0; j < h; j++)
                {
                    float recurrent = values[(gates + gate) * h + j];
                    if (!lstm && gate == 2 && recurrent != 0f)
                    {
                        throw Unsupported(rnn, "a GRU with a recurrent bias on the candidate gate");
                    }

                    bias[order[gate] * h + j] = values[gate * h + j] + recurrent;
                }
            }
        }

        var (inputWeight, hiddenWeight) = (Regroup(w.AsFloats(), inputs), Regroup(rw.AsFloats(), h));
        var nodes = new List<OnnxNode> { timeMajor, rnn };
        bool sequences;
        string output;
        if (Users(rnn.Outputs[0]) is [{ Op: "Squeeze" } squeeze] && SqueezeAxes(squeeze) is [1]
            && Only(squeeze.Outputs[0], "Transpose") is { } back && back.Ints("perm") is [1, 0, 2])
        {
            nodes.AddRange([squeeze, back]);
            (sequences, output) = (true, back.Outputs[0]);
        }
        else if (rnn.Outputs.Count > 1 && Users(rnn.Outputs[1]) is [{ Op: "Squeeze" } last] && SqueezeAxes(last) is [0])
        {
            nodes.Add(last);
            (sequences, output) = (false, last.Outputs[0]);
        }
        else
        {
            throw Unsupported(rnn, "a recurrent layer whose outputs are not used as [batch, time, hidden] or the last state");
        }

        Consume(nodes);
        return Push(builder => lstm ? builder.LSTM(h, sequences) : builder.GRU(h, sequences), m =>
        {
            var module = (RecurrentModule)m;
            module.InputWeight.Load(inputWeight);
            module.HiddenWeight.Load(hiddenWeight);
            module.Bias.Load(bias);
        }, output);
    }

    private long[]? SqueezeAxes(OnnxNode squeeze) => squeeze.Ints("axes") ?? (squeeze.Inputs.Count > 1 ? Const(squeeze.Inputs[1])?.AsLongs() : null);

    // ------------------------------------------------------------------ the rest

    private string Conv(OnnxNode node, OnnxTensor weight, OnnxTensor? bias)
    {
        var (k, stride, padding) = Window(node);
        if (node.Int("group", 1) != 1 || weight.Dims is not [var outC, _, var kh, var kw] || kh != kw || kh != k)
        {
            throw Unsupported(node, "grouped or non-square convolution");
        }

        return Push(b => b.Conv2d(outC, k, stride, padding, bias is not null), m =>
        {
            var conv = (Conv2d)m;
            conv.Weight.Load(weight.AsFloats());                                       // [out, in, k, k] = [out, in·k·k]
            if (bias is not null)
            {
                conv.Bias!.Load(bias.AsFloats());
            }
        }, node.Outputs[0]);
    }

    private static (int Kernel, int Stride, int Padding) Window(OnnxNode node)
    {
        var kernel = node.Ints("kernel_shape");
        var strides = node.Ints("strides") ?? [1, 1];
        var pads = node.Ints("pads") ?? [0, 0, 0, 0];
        var dilations = node.Ints("dilations") ?? [1, 1];
        if (node.String("auto_pad") is { } autoPad && autoPad != "NOTSET" || kernel is not [var kh, var kw] || kh != kw
            || strides is not [var sh, var sw] || sh != sw || pads.Distinct().Count() != 1 || dilations.Any(d => d != 1))
        {
            throw Unsupported(node, "a 2-D window with a square kernel, equal strides, equal padding on all sides and no dilation");
        }

        return ((int)kh, (int)sh, (int)pads[0]);
    }

    private string Gather(OnnxNode node, string x)
    {
        if (node.Inputs[1] == x && Const(node.Inputs[0]) is { Dims.Length: 2 } table)
        {
            return Push(b => b.Embedding(table.Dims[0], table.Dims[1]), m => ((Embedding)m).Weight.Load(table.AsFloats()), node.Outputs[0]);
        }

        if (node.Inputs[0] == x && node.Int("axis", 0) == 1 && Const(node.Inputs[1]) is { Dims.Length: 0 } index && _network.CurrentShape.Count == 2)
        {
            long i = index.AsLongs()[0], t = _network.CurrentShape[0];
            if (i == 0)
            {
                return Push(b => b.FirstStep(), null, node.Outputs[0]);
            }

            if (i == t - 1 || i == -1)
            {
                return Push(b => b.LastStep(), null, node.Outputs[0]);
            }
        }

        throw Unsupported(node, "a gather that is not an embedding lookup or the first/last time step");
    }

    private string PositionTable(OnnxNode node, string x)
    {
        if (Const(Other(node, x) ?? "") is { Dims: [var t, var d] } table && _network.CurrentShape.SequenceEqual([t, d]))
        {
            var values = table.AsFloats();
            for (int pos = 0; pos < t; pos++)
            {
                for (int i = 0; i < d; i++)
                {
                    double angle = pos / Math.Pow(10000, 2 * (i / 2) / (double)d);
                    if (MathF.Abs(values[pos * d + i] - (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle))) > 1e-4f)
                    {
                        throw Unsupported(node, "a learned (non-sinusoidal) position table");
                    }
                }
            }

            return Push(b => b.PositionalEncoding(), null, node.Outputs[0]);
        }

        throw Unsupported(node, "adding a value that is not a position table");
    }
}
