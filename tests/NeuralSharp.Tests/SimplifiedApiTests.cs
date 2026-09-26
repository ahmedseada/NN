using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Diagnostics;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;
using NeuralSharp.Training;

// Each simplified API must behave exactly like the code it replaces: same layers, same weights, same training results.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Simplified =
    [
        ("builder: same layers, shapes and weights as new Sequential { ... }", BuilderMatchesSequential),
        ("builder: CNN, RNN and GPT shapes; JSON round trip; clear shape errors", BuilderShapesAndJson),
        ("data: Split/Standardize/Batches equal the manual steps", DataExtensionsMatchManual),
        ("training: factory Trainer and TrainingRun give the same history as the manual trainer", TrainingRunMatchesTrainer),
        ("training: FitAsync progress and TrainAsync early exit", TrainingAsync),
        ("fine-tuning: freeze, trainable-only save/load, grouped optimizer", FreezingAndGroups),
        ("fine-tuning: LoRA starts neutral, trains only adapters, merges exactly", Lora),
    ];

    private static Sequential ManualMlp(Device device, Random r) => new()
    {
        new Linear(9, 16, device: device, random: r), new ReLU(), new Dropout(0.1f, r),
        new Linear(16, 8, device: device, random: r), new Tanh(),
        new Linear(8, 1, device: device, random: r),
    };

    private static NetworkBuilder BuiltMlp(Device device) =>
        Network.Input(9).OnDevice(device).Seed(7).Linear(16).ReLU().Dropout(0.1f).Linear(8).Tanh().Linear(1);

    private static void BuilderMatchesSequential(Device device)
    {
        using var manual = ManualMlp(device, new Random(7));
        using var built = BuiltMlp(device).Build();
        Check(built.Count == manual.Count, "layer count");
        for (int i = 0; i < manual.Count; i++)
        {
            Check(built[i].ToString() == manual[i].ToString(), $"layer {i}: {built[i]} vs {manual[i]}");
        }

        var a = manual.Parameters().SelectMany(p => p.ToArray()).ToArray();
        var b = built.Parameters().SelectMany(p => p.ToArray()).ToArray();
        Check(a.SequenceEqual(b), "same seed must give identical initial weights");

        var x = new float[4, 9];
        for (int i = 0; i < 36; i++) x[i / 9, i % 9] = i * 0.1f - 1.5f;
        Check(manual.Predict(x).Cast<float>().SequenceEqual(built.Predict(x).Cast<float>()), "same outputs");
    }

    private static void BuilderShapesAndJson(Device device)
    {
        var cnn = Architectures.Cnn(1, 16, 16, [16, 32], 3, 4).OnDevice(device).Seed(1);
        Check(cnn.CurrentShape.SequenceEqual([4]), $"cnn output {string.Join(",", cnn.CurrentShape)}");
        using (var model = cnn.Build())
        using (var input = Tensor.Zeros([2, 1, 16, 16], device))
        using (var output = model.Predict(input))
        {
            Check(output.Shape.SequenceEqual([2, 4]), "cnn forward shape");
            Check(model.OfType<Flatten>().Any(), "flatten present");
        }

        var gpt = Architectures.Gpt(vocabulary: 20, context: 12, dim: 16, heads: 4, layers: 2, ffDim: 32, dropout: 0f).OnDevice(device).Seed(3);
        var json = gpt.ToJson();
        var replayed = Network.FromJson(json).OnDevice(device);
        Check(replayed.ToJson().ToJsonString() == json.ToJsonString(), "JSON round trip");
        using (var a = gpt.Build())
        using (var b = replayed.Build())
        {
            Check(a.Parameters().SelectMany(p => p.ToArray()).SequenceEqual(b.Parameters().SelectMany(p => p.ToArray())), "replayed weights");
            Check(Network.ArchitectureOf(a)?.ToJsonString() == json.ToJsonString(), "architecture registered");
            using var ids = Tensor.Zeros([2, 12], device);
            using var logits = a.Predict(ids);
            Check(logits.Shape.SequenceEqual([2, 12, 20]), "gpt forward shape");
        }

        var rnn = Architectures.Rnn(RecurrentCell.GRU, vocabulary: 10, length: 6, embed: 8, hidden: 12, outputs: 2);
        Check(rnn.CurrentShape.SequenceEqual([2]), "rnn output");
        bool threw = false;
        try { Network.Input(4).Conv2d(8, 3); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "Conv2d after a feature input must fail with a clear error");
        threw = false;
        try { Network.Input(4).Linear(8, inputs: 5); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "explicit inputs are checked");
        var custom = Network.Input(4).Add(new Linear(4, 3), [3]);
        Check(!custom.IsDescribable, "a custom module makes the builder non-describable");
    }

    private static Dataset SmallRegression(int count, int seed)
    {
        var r = new Random(seed);
        var x = new float[count, 3];
        var y = new float[count, 1];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < 3; j++) x[i, j] = r.NextSingle() * 10 + j * 5;
            y[i, 0] = 2 * x[i, 0] - x[i, 1] + 0.5f * x[i, 2] + r.NextSingle();
        }

        return Dataset.FromArrays(x, y);
    }

    private static void DataExtensionsMatchManual(Device device)
    {
        var data = SmallRegression(100, 1);
        var (train, test) = data.Split(0.8, seed: 2);
        var fs = StandardScaler.FitFeatures(train);
        var ts = StandardScaler.FitTargets(train);
        var trainScaled = train.Scale(fs, ts);
        var testScaled = test.Scale(fs, ts);

        var split = data.Split(0.8, seed: 2).StandardizeFeatures().StandardizeTargets();
        Check(split.Train.Features.SequenceEqual(trainScaled.Features) && split.Train.Targets.SequenceEqual(trainScaled.Targets), "train part");
        Check(split.Test.Features.SequenceEqual(testScaled.Features) && split.Test.Targets.SequenceEqual(testScaled.Targets), "test part");
        Check(split.FeatureScaler is StandardScaler f && f.Mean.SequenceEqual(fs.Mean), "feature scaler kept");

        var loader = split.Train.Batches(16, shuffle: true, device: device, seed: 4);
        Check(loader.BatchSize == 16 && loader.Shuffle && loader.Device == device && loader.BatchCount == 5, "Batches = DataLoader");
        var normalized = data.Split(0.8, seed: 2).NormalizeFeatures();
        Check(normalized.Train.Features.ToArray().All(v => v >= -1e-6f && v <= 1 + 1e-6f), "min-max scaled into [0, 1]");

        var path = Path.GetTempFileName();
        ((MinMaxScaler)normalized.FeatureScaler!).Save(path);
        var loaded = MinMaxScaler.Load(path);
        File.Delete(path);
        Check(loaded.Min.SequenceEqual(((MinMaxScaler)normalized.FeatureScaler).Min), "MinMaxScaler save/load");
    }

    private static TrainingHistory ManualFit(Device device, DataSplit split)
    {
        using var model = BuiltMlpFor3(device);
        using var optimizer = new AdamW(model.Parameters(), 0.01f, weightDecay: 1e-4f);
        var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
        {
            Metrics = { Metric.MeanAbsoluteError },
            Scheduler = new CosineAnnealing(optimizer, 6, warmupEpochs: 1),
            MaxGradientNorm = 1f,
        };
        return trainer.Fit(new DataLoader(split.Train, 16, shuffle: true, device: device, seed: 5), 6,
            validation: new DataLoader(split.Test, 64, device: device));
    }

    private static Sequential BuiltMlpFor3(Device device) =>
        Network.Input(3).OnDevice(device).Seed(11).Linear(16).ReLU().Linear(1).Build();

    private static void TrainingRunMatchesTrainer(Device device)
    {
        var split = SmallRegression(200, 3).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        var manual = ManualFit(device, split);

        using var model = BuiltMlpFor3(device);
        var history = new TrainingRun
        {
            Model = model,
            Loss = Losses.MeanSquaredError,
            Optimizer = p => new AdamW(p, 0.01f, weightDecay: 1e-4f),
            Scheduler = o => new CosineAnnealing(o, 6, warmupEpochs: 1),
            Train = split.Train.Batches(16, shuffle: true, device: device, seed: 5),
            Validation = split.Test.Batches(64, device: device),
            Epochs = 6,
            Metrics = [Metric.MeanAbsoluteError],
            MaxGradientNorm = 1f,
        }.Fit();

        Check(history.Epochs.Count == manual.Epochs.Count, "epoch count");
        for (int i = 0; i < manual.Epochs.Count; i++)
        {
            Check(Math.Abs(history.Epochs[i].Loss - manual.Epochs[i].Loss) < 1e-9 &&
                  Math.Abs(history.Epochs[i].ValidationLoss!.Value - manual.Epochs[i].ValidationLoss!.Value) < 1e-9 &&
                  history.Epochs[i].LearningRate == manual.Epochs[i].LearningRate,
                $"epoch {i + 1}: {history.Epochs[i].Loss} vs {manual.Epochs[i].Loss}");
        }

        using var model2 = BuiltMlpFor3(device);
        using var trainer = new Trainer(model2, Losses.MeanSquaredError, p => new AdamW(p, 0.01f, weightDecay: 1e-4f),
            o => new CosineAnnealing(o, 6, warmupEpochs: 1)) { MaxGradientNorm = 1f, Metrics = { Metric.MeanAbsoluteError } };
        var h2 = trainer.Fit(split.Train.Batches(16, shuffle: true, device: device, seed: 5), 6, split.Test.Batches(64, device: device));
        Check(Math.Abs(h2.Epochs[^1].Loss - manual.Epochs[^1].Loss) < 1e-9, "factory constructor");
    }

    private static void TrainingAsync(Device device)
    {
        var split = SmallRegression(120, 5).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        using var model = BuiltMlpFor3(device);
        var run = new TrainingRun
        {
            Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
            Train = split.Train.Batches(16, device: device), Epochs = 5,
        };
        var seen = new List<int>();
        var progress = new SyncProgress<EpochCompleted>(e => seen.Add(e.Epoch));
        var history = run.FitAsync(progress).GetAwaiter().GetResult();
        Check(history.Epochs.Count == 5 && seen.SequenceEqual([1, 2, 3, 4, 5]), $"progress {string.Join(",", seen)}");

        using var model2 = BuiltMlpFor3(device);
        var epochs = new List<int>();
        Task.Run(async () =>
        {
            await foreach (var e in (run with { Model = model2, Epochs = 50 }).TrainAsync())
            {
                epochs.Add(e.Epoch);
                if (e.Epoch == 3) break;
            }
        }).GetAwaiter().GetResult();
        Check(epochs.SequenceEqual([1, 2, 3]), $"early exit {string.Join(",", epochs)}");
    }

    private sealed class SyncProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    private static void FreezingAndGroups(Device device)
    {
        using var model = Network.Input(3).OnDevice(device).Seed(2).Linear(8).ReLU().Linear(8).ReLU().Linear(1).Build();
        model.Freeze(0..^1);
        Check(model.TrainableParameters().Count() == 2 && model[^1].Parameters().All(p => p.RequiresGrad), "only the head trains");
        var before = model[0].Parameters().First().ToArray();
        var split = SmallRegression(64, 9).Split(0.75, seed: 1).StandardizeFeatures().StandardizeTargets();
        using (var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.05f)))
        {
            Check(trainer.Optimizer.Parameters.Count == 2, "factory receives trainable parameters only");
            trainer.Fit(split.Train.Batches(16, device: device), 3);
        }

        Check(model[0].Parameters().First().ToArray().SequenceEqual(before), "frozen layer unchanged");
        var path = Path.GetTempFileName();
        model.SaveTrainable(path);
        using var other = Network.Input(3).OnDevice(device).Seed(99).Linear(8).ReLU().Linear(8).ReLU().Linear(1).Build();
        other.Freeze(0..^1);
        other.LoadTrainable(path);
        Check(other[^1].Parameters().First().ToArray().SequenceEqual(model[^1].Parameters().First().ToArray()), "head restored");
        Check(new FileInfo(path).Length < 200, "only the head was saved");
        File.Delete(path);
        model.Unfreeze();
        Check(model.TrainableParameters().Count() == 6, "unfreeze");

        using var body = Network.Input(3).OnDevice(device).Seed(4).Linear(4).Build();
        using var head = Network.Input(4).OnDevice(device).Seed(5).Linear(1).Build();
        using var grouped = new GroupedOptimizer(new Sgd(body.Parameters(), 0.01f), new Sgd(head.Parameters(), 0.1f));
        grouped.LearningRate = 0.005f;                 // a schedule halving the rate
        grouped.Step();
        Check(Math.Abs(grouped.Groups[0].LearningRate - 0.005f) < 1e-9 && Math.Abs(grouped.Groups[1].LearningRate - 0.05f) < 1e-7, "ratio kept");
    }

    private static void Lora(Device device)
    {
        using var model = Architectures.Gpt(vocabulary: 12, context: 8, dim: 16, heads: 2, layers: 1, ffDim: 32, dropout: 0f).OnDevice(device).Seed(8).Build();
        var ids = new float[2 * 8];
        for (int i = 0; i < ids.Length; i++) ids[i] = i % 12;
        using var x = Tensor.From(ids, [2, 8], device);
        float[] before;
        using (var y = model.Predict(x)) before = y.ToArray();

        int added = model.AddLora(rank: 4, alpha: 8, targets: _ => true, freezeBase: true, random: new Random(1));
        Check(added == 5, $"adapters on qkv, output, ff1, ff2 and head: {added}");
        using (var y = model.Predict(x)) Check(y.ToArray().SequenceEqual(before), "B = 0: outputs unchanged");
        Check(model.TrainableParameters().Count() == 10, "only A and B train");

        var targets = new float[2 * 8];
        for (int i = 0; i < targets.Length; i++) targets[i] = (i + 1) % 12;
        using var t = Tensor.From(targets, [2, 8], device);
        using (var trainer = new Trainer(model, (p, y) => Losses.SparseCrossEntropy(p, y), p => new Adam(p, 0.05f)))
        {
            for (int step = 0; step < 5; step++)
            {
                using var scope = new TensorScope();
                trainer.Optimizer.ZeroGrad();
                Losses.SparseCrossEntropy(model.Forward(x), t).Backward();
                trainer.Optimizer.Step();
            }
        }

        float[] adapted;
        using (var y = model.Predict(x)) adapted = y.ToArray();
        Check(!adapted.SequenceEqual(before), "training changed the outputs");
        int merged = model.MergeLora();
        using (var y = model.Predict(x))
        {
            var after = y.ToArray();
            Check(merged == 5 && after.Zip(adapted).All(p => MathF.Abs(p.First - p.Second) < 1e-4f), "merge keeps outputs");
        }

        Check(model.Parameters().Count() == model.Descendants().OfType<Linear>().Sum(l => l.Bias is null ? 1 : 2)
              + model.Descendants().OfType<LayerNorm>().Count() * 2 + 1, "adapters removed");
    }
}
