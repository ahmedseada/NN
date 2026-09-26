using NeuralSharp;
using NeuralSharp.Inference;
using NeuralSharp.Layers;
using NeuralSharp.Onnx;
using NeuralSharp.Onnx.Runtime;

// ONNX export: every supported layer is exported, run by ONNX Runtime and compared with NeuralSharp's own output.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Onnx =
    [
        ("onnx: MLP with BatchNorm, GELU, Tanh, Sigmoid, Dropout, Softmax matches ONNX Runtime", OnnxMlp),
        ("onnx: CNN (Conv2d with stride and padding, BatchNorm, MaxPool2d, GlobalAveragePool2d, Flatten) matches", OnnxCnn),
        ("onnx: LSTM and GRU (sequences and last state), LastStep, FirstStep, Reshape match", OnnxRecurrent),
        ("onnx: transformer classifier and causal GPT (Embedding, PositionalEncoding, attention) match", OnnxTransformer),
        ("onnx: LoRA merged on export, custom lambda translator, metadata, names, errors, predictor", OnnxExtras),
        ("onnx import: MLP, CNN, LSTM/GRU, transformer, GPT round-trip into NeuralSharp layers on the device", OnnxImportRoundTrip),
        ("onnx import: PyTorch-style Gemm and erf GELU; .nsm package; unsupported ops are named", OnnxImportForeign),
    ];

    // Export → import into NeuralSharp layers (on `device`) → the same outputs as the original.
    private static void CheckImport(Module model, int[] sampleShape, Tensor input, string what, Func<OnnxExporter, OnnxExporter>? configure = null)
    {
        model.Eval();
        var exporter = OnnxExport.For(model).Input(sampleShape);
        using var imported = OnnxImport.Load((configure?.Invoke(exporter) ?? exporter).ToBytes(), input.Device);
        using var expected = model.Predict(input);
        using var actual = imported.Model.Predict(input);
        Check(actual.Shape.SequenceEqual(expected.Shape), $"{what}: shape {Tensor.FormatShape(actual.Shape)} vs {Tensor.FormatShape(expected.Shape)}");
        float worst = actual.ToArray().Zip(expected.ToArray(), (x, y) => MathF.Abs(x - y)).Max();
        Check(worst < 1e-4f, $"{what}: largest difference {worst}");
    }

    private static void OnnxImportRoundTrip(Device device)
    {
        var r = new Random(20);
        using var mlp = Network.Input(6).OnDevice(device).Seed(2).Linear(16).BatchNorm().GELU().Dropout(0.3f).Linear(12).Tanh()
            .Linear(8).Sigmoid().Linear(8).ReLU().Linear(5, bias: false).Softmax().Build();
        WarmBatchNorm(mlp, RandomInput(device, r, 32, 6) * 3f);
        CheckImport(mlp, [6], RandomInput(device, r, 4, 6), "mlp");

        using var cnn = Network.Image(2, 12, 12).OnDevice(device).Seed(4).Conv2d(4, 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)
            .Conv2d(6, 3, stride: 2, padding: 1).ReLU().GlobalAveragePool2d().Linear(3).Build();
        WarmBatchNorm(cnn, RandomInput(device, r, 16, 2, 12, 12));
        CheckImport(cnn, [2, 12, 12], RandomInput(device, r, 3, 2, 12, 12), "cnn");
        using var flat = Network.Image(1, 8, 8).OnDevice(device).Seed(5).Conv2d(3, 3, bias: false).Flatten().Linear(4).Build();
        CheckImport(flat, [1, 8, 8], RandomInput(device, r, 2, 1, 8, 8), "cnn with flatten");

        using var rnn = Network.Sequence(7, 4).OnDevice(device).Seed(7).LSTM(6, returnSequences: true).GRU(5, returnSequences: true).LastStep().Linear(2).Build();
        CheckImport(rnn, [7, 4], RandomInput(device, r, 3, 7, 4), "lstm → gru → last step");
        using var last = Network.Sequence(5, 3).OnDevice(device).Seed(9).LSTM(4).Reshape(2, 2).Build();
        CheckImport(last, [5, 3], RandomInput(device, r, 2, 5, 3), "lstm last state → reshape");
        using var first = Network.Sequence(5, 3).OnDevice(device).Seed(10).GRU(4).Build();
        CheckImport(first, [5, 3], RandomInput(device, r, 2, 5, 3), "gru last state");

        using var classifier = Architectures.TransformerClassifier(vocabulary: 20, length: 10, dim: 16, heads: 4, layers: 2, ffDim: 32, dropout: 0.1f, outputs: 3)
            .OnDevice(device).Seed(12).Build();
        CheckImport(classifier, [10], RandomIds(device, r, 20, 3, 10), "transformer classifier");
        using var gpt = Architectures.Gpt(vocabulary: 24, context: 12, dim: 16, heads: 2, layers: 2, ffDim: 32, dropout: 0.1f).OnDevice(device).Seed(13).Build();
        CheckImport(gpt, [12], RandomIds(device, r, 24, 2, 12), "causal gpt");
        using var attention = Network.Sequence(6, 8).OnDevice(device).Seed(14).MultiHeadAttention(heads: 2).LayerNorm().Build();
        CheckImport(attention, [6, 8], RandomInput(device, r, 2, 6, 8), "attention alone");
        using var cls = Network.Sequence(5, 3).OnDevice(device).Seed(21).GRU(4, returnSequences: true).FirstStep().Build();
        CheckImport(cls, [5, 3], RandomInput(device, r, 2, 5, 3), "first step");
    }

    private static void OnnxImportForeign(Device device)
    {
        var r = new Random(22);
        using var mlp = Network.Input(5).OnDevice(device).Seed(23).Linear(8).GELU().Linear(3).Softmax().Build();
        mlp.Eval();
        // Written the way PyTorch exports nn.Linear (Gemm with a transposed [out, in] weight) and exact GELU (erf).
        var bytes = OnnxExport.For(mlp).Input(5)
            .Module<Linear>((g, linear, x, shape) => g.Node("Gemm",
                [x, g.Constant("w", TransposeRows(linear.Weight.ToArray(), linear.InFeatures, linear.OutFeatures), linear.OutFeatures, linear.InFeatures),
                    g.Constant("b", linear.Bias!.ToArray(), linear.OutFeatures)], shape, OnnxAttribute.Of("transB", 1L)))
            .Module<GELU>((g, _, x, shape) => g.Node("Mul", [g.Node("Mul", [x, g.Node("Add", [g.Node("Erf", [g.Node("Div", [x, g.Scalar("sqrt2", MathF.Sqrt(2f))])]),
                g.Scalar("one", 1f)])]), g.Scalar("half", 0.5f)], shape))
            .Metadata("source", "pytorch-style").ToBytes();
        using var imported = OnnxImport.Load(bytes, device);
        Check(imported.Model.Select(m => m.GetType().Name).SequenceEqual(["Linear", "GELU", "Linear", "Softmax"]), string.Join(", ", imported.Model.Select(m => m.ToString())));
        Check(imported.Notes.Count == 1 && imported.Notes[0].Contains("erf"), "the erf → tanh approximation is reported");
        Check(imported.Metadata["source"] == "pytorch-style", "metadata");
        var x = RandomInput(device, r, 4, 5);
        using var expected = mlp.Predict(x);
        using var actual = imported.Model.Predict(x);
        Check(actual.ToArray().Zip(expected.ToArray(), (a, b) => MathF.Abs(a - b)).Max() < 1e-5f, "gemm with transposed weights");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.nsm");
        try
        {
            imported.SavePackage(path);
            using var predictor = Predictor.Load(path, device).Build();
            var p = predictor.Predict([0.1f, -0.2f, 0.3f, 0.4f, -0.5f]);
            using var single = Tensor.From([0.1f, -0.2f, 0.3f, 0.4f, -0.5f], [1, 5], device);
            using var reference = mlp.Predict(single);
            Check(p.Zip(reference.ToArray(), (a, b) => MathF.Abs(a - b)).Max() < 1e-5f, "the .nsm package predicts the same");
        }
        finally
        {
            File.Delete(path);
        }

        using var custom = new Sequential { new Linear(4, 4, device: device, random: new Random(24)), new Lambda(x => x * 2f + 1f, "Affine") };
        var unsupported = OnnxExport.For(custom).Input(4)
            .Lambda("Affine", (g, _, v, shape) => g.Node("Add", [g.Node("Mul", [v, g.Scalar("two", 2f)]), g.Scalar("one", 1f)], shape)).ToBytes();
        try
        {
            OnnxImport.Load(unsupported, device).Dispose();
            Check(false, "an unsupported operator must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Mul"), ex.Message);
        }
    }

    private static float[] TransposeRows(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[j * rows + i] = values[i * columns + j];
            }
        }

        return result;
    }

    private static void CheckOnnx(Module model, int[] sampleShape, Tensor input, string what, Func<OnnxExporter, OnnxExporter>? configure = null)
    {
        model.Eval();
        var exporter = OnnxExport.For(model).Input(sampleShape);
        using var onnx = OnnxModule.Load((configure?.Invoke(exporter) ?? exporter).ToBytes());
        using var expected = model.Predict(input);
        using var actual = onnx.Predict(input);
        Check(actual.Shape.SequenceEqual(expected.Shape), $"{what}: shape {Tensor.FormatShape(actual.Shape)} vs {Tensor.FormatShape(expected.Shape)}");
        var (a, e) = (actual.ToArray(), expected.ToArray());
        float worst = a.Zip(e, (x, y) => MathF.Abs(x - y)).Max();
        Check(worst < 1e-4f, $"{what}: largest difference {worst}");
    }

    private static Tensor RandomInput(Device device, Random r, params int[] shape) =>
        Tensor.From([.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(r.NextDouble() * 2 - 1))], shape, device);

    private static Tensor RandomIds(Device device, Random r, int vocabulary, params int[] shape) =>
        Tensor.From([.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)r.Next(vocabulary))], shape, device);

    // Runs a few training-mode batches so BatchNorm's running statistics are not the initial 0 / 1.
    private static void WarmBatchNorm(Module model, Tensor input)
    {
        model.Train();
        using (Autograd.NoGrad())
        {
            for (int i = 0; i < 3; i++)
            {
                model.Forward(input).Dispose();
            }
        }

        model.Eval();
    }

    private static void OnnxMlp(Device device)
    {
        var r = new Random(1);
        using var mlp = Network.Input(6).OnDevice(device).Seed(2).Linear(16).BatchNorm().GELU().Dropout(0.3f).Linear(12).Tanh()
            .Linear(8).Sigmoid().Linear(8).ReLU().Linear(5, bias: false).Softmax().Build();
        var x = RandomInput(device, r, 4, 6);
        WarmBatchNorm(mlp, RandomInput(device, r, 32, 6) * 3f);
        CheckOnnx(mlp, [6], x, "mlp");
    }

    private static void OnnxCnn(Device device)
    {
        var r = new Random(3);
        using var cnn = Network.Image(2, 12, 12).OnDevice(device).Seed(4)
            .Conv2d(4, 3, padding: 1).BatchNorm().ReLU().MaxPool2d(2)
            .Conv2d(6, 3, stride: 2, padding: 1).ReLU().MaxPool2d(2, stride: 1, padding: 1)
            .GlobalAveragePool2d().Linear(3).Build();
        WarmBatchNorm(cnn, RandomInput(device, r, 16, 2, 12, 12));
        CheckOnnx(cnn, [2, 12, 12], RandomInput(device, r, 3, 2, 12, 12), "cnn with global pooling");
        using var flat = Network.Image(1, 8, 8).OnDevice(device).Seed(5).Conv2d(3, 3, bias: false).Flatten().Linear(4).Build();
        CheckOnnx(flat, [1, 8, 8], RandomInput(device, r, 2, 1, 8, 8), "cnn with flatten");
    }

    private static void OnnxRecurrent(Device device)
    {
        var r = new Random(6);
        using var lstm = Network.Sequence(7, 4).OnDevice(device).Seed(7).LSTM(6, returnSequences: true).GRU(5, returnSequences: true).LastStep().Linear(2).Build();
        CheckOnnx(lstm, [7, 4], RandomInput(device, r, 3, 7, 4), "lstm → gru → last step");
        using var last = Network.Sequence(5, 3).OnDevice(device).Seed(8).GRU(4).Linear(2).Build();
        CheckOnnx(last, [5, 3], RandomInput(device, r, 2, 5, 3), "gru last state");
        using var lstmLast = Network.Sequence(5, 3).OnDevice(device).Seed(9).LSTM(4).Reshape(2, 2).Build();
        CheckOnnx(lstmLast, [5, 3], RandomInput(device, r, 2, 5, 3), "lstm last state → reshape");
        using var first = Network.Sequence(5, 3).OnDevice(device).Seed(10).GRU(4, returnSequences: true).FirstStep().Build();
        CheckOnnx(first, [5, 3], RandomInput(device, r, 2, 5, 3), "first step");
    }

    private static void OnnxTransformer(Device device)
    {
        var r = new Random(11);
        using var classifier = Architectures.TransformerClassifier(vocabulary: 20, length: 10, dim: 16, heads: 4, layers: 2, ffDim: 32, dropout: 0.1f, outputs: 3)
            .OnDevice(device).Seed(12).Build();
        CheckOnnx(classifier, [10], RandomIds(device, r, 20, 3, 10), "transformer classifier");
        using var gpt = Architectures.Gpt(vocabulary: 24, context: 12, dim: 16, heads: 2, layers: 2, ffDim: 32, dropout: 0.1f).OnDevice(device).Seed(13).Build();
        CheckOnnx(gpt, [12], RandomIds(device, r, 24, 2, 12), "causal gpt");
        using var attention = Network.Sequence(6, 8).OnDevice(device).Seed(14).MultiHeadAttention(heads: 2).LayerNorm().Build();
        CheckOnnx(attention, [6, 8], RandomInput(device, r, 2, 6, 8), "attention alone");
    }

    private static void OnnxExtras(Device device)
    {
        var r = new Random(15);
        using var tuned = Network.Input(5).OnDevice(device).Seed(16).Linear(8).ReLU().Linear(3).Build();
        tuned.AddLora(rank: 2, alpha: 4, targets: _ => true, freezeBase: true, random: new Random(17));
        foreach (var linear in tuned.OfType<Linear>())
        {
            linear.Adapter!.B.Load([.. Enumerable.Range(0, linear.Adapter.B.Size).Select(i => MathF.Sin(i))]);   // a non-zero update
        }

        CheckOnnx(tuned, [5], RandomInput(device, r, 3, 5), "lora");

        using var custom = new Sequential { new Linear(4, 4, device: device, random: new Random(18)), new Lambda(x => x * 2f + 1f, "Affine") };
        CheckOnnx(custom, [4], RandomInput(device, r, 2, 4), "custom lambda", e => e.Lambda("Affine",
            (g, _, x, shape) => g.Node("Add", [g.Node("Mul", [x, g.Scalar("two", 2f)]), g.Scalar("one", 1f)], shape)));
        try
        {
            OnnxExport.For(custom).Input(4).ToBytes();
            Check(false, "an unknown lambda must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Affine"), ex.Message);
        }

        using var small = Network.Input(3).OnDevice(device).Seed(19).Linear(2).Softmax().Build();
        using var onnx = OnnxModule.Load(OnnxExport.For(small).Input(3).Names("features", "probabilities").Metadata("classes", "cat,dog").ToBytes());
        Check(onnx.InputName == "features" && onnx.OutputName == "probabilities", $"{onnx.InputName} {onnx.OutputName}");
        Check(onnx.InputShape.SequenceEqual([-1, 3]) && onnx.OutputShape.SequenceEqual([-1, 2]), onnx.ToString());
        Check(onnx.Metadata.TryGetValue("classes", out var classes) && classes == "cat,dog", "metadata");

        using var predictor = Predictor.For(onnx).Classes(["cat", "dog"]).Build();
        using var reference = Predictor.For(small).Classes(["cat", "dog"]).Build();
        var p = predictor.Predict([0.5f, -1f, 2f]);
        var q = reference.Predict([0.5f, -1f, 2f]);
        Check(p.Class == q.Class && MathF.Abs(p.Probability - q.Probability) < 1e-5f, "predictor over an ONNX model");

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.onnx");
        try
        {
            small.ExportOnnx(path, 3);
            using var fromFile = OnnxModule.Load(path);
            Check(fromFile.InputName == "input" && fromFile.OutputName == "output", "default names");
        }
        finally
        {
            File.Delete(path);
        }
    }
}
