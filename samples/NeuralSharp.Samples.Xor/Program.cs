// Learns XOR, the classic problem a single linear layer cannot solve.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Xor               GPU if available, else CPU
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Xor -- --cpu      force the CPU
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Xor -- --cuda     force the GPU
//   dotnet run -c Release --project samples/NeuralSharp.Samples.Xor -- --help     all options

using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Diagnostics;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;
using NeuralSharp.Samples;
using NeuralSharp.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");

// Telemetry: print one line every 250 epochs (plus the first and last), and optionally log everything to a file.
using var console = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 250));
await using var fileLog = options.LogFile is { } path ? new JsonLinesLogger(path, TelemetryLevel.All) : null;
using var fileSubscription = fileLog is null ? null : Telemetry.Subscribe(fileLog);

// The four XOR cases: the output is 1 when exactly one input is 1.
var data = Dataset.FromArrays(
    features: new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    targets: new float[,] { { 0 }, { 1 }, { 1 }, { 0 } },
    featureNames: ["a", "b"],
    targetNames: ["a xor b"]);

// 2 inputs -> 8 hidden units (tanh) -> 1 output (sigmoid, so it reads as a probability).
var random = new Random(42);
using var model = new Sequential
{
    new Linear(2, 8, device: device, random: random),
    new Tanh(),
    new Linear(8, 1, device: device, random: random),
    new Sigmoid(),
};
model.Name = "xor";

using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);
var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
};

var loader = new DataLoader(data, batchSize: options.BatchSize ?? 4, device: device);
trainer.Fit(loader, epochs: options.Epochs ?? 2000);

// Inference.
var predictions = model.Predict(data.FeaturesToArray());
var x = data.FeaturesToArray();
var y = data.TargetsToArray();
int correct = 0;

Console.WriteLine("\n A  B | expected | predicted | rounded");
Console.WriteLine("------+----------+-----------+--------");
for (int i = 0; i < data.Count; i++)
{
    int rounded = predictions[i, 0] >= 0.5f ? 1 : 0;
    bool ok = rounded == (int)y[i, 0];
    correct += ok ? 1 : 0;
    Console.WriteLine($" {x[i, 0]}  {x[i, 1]} |    {y[i, 0]}     |  {predictions[i, 0]:F4}   |   {rounded} {(ok ? "✓" : "✗")}");
}

Console.WriteLine($"\n{correct}/{data.Count} correct");
return correct == data.Count ? 0 : 1;
