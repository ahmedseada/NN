// Imports an ONNX model made by another framework into NeuralSharp layers and runs it on NeuralSharp's own
// backend (CPU or CUDA). With a "<file>.expected.json" next to the model (inputs and the other framework's outputs,
// as written by tools/pytorch/xor_to_onnx.py), it checks that NeuralSharp computes the same values, then saves the
// model as a NeuralSharp package (.nsm), reloads it and checks it again.
//
//   python tools/pytorch/xor_to_onnx.py --out xor.onnx
//   dotnet run -c Release --project samples/NeuralSharp.Samples.OnnxImport -- --onnx xor.onnx --cuda

using System.Diagnostics;
using System.Text.Json.Nodes;
using NeuralSharp;
using NeuralSharp.Inference;
using NeuralSharp.Onnx;
using NeuralSharp.Samples;

if (SampleOptions.Parse(args, ("onnx", "the .onnx file to import (required)"), ("tolerance", "largest allowed difference (default 1e-5)")) is not { } options)
{
    return 0;
}

string path = options.Get("onnx") ?? throw new ArgumentException("Pass --onnx <file>.");
float tolerance = float.Parse(options.Get("tolerance", "1e-5")!, System.Globalization.CultureInfo.InvariantCulture);
var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}");

var clock = Stopwatch.StartNew();
using var imported = OnnxImport.Load(path, device);
Console.WriteLine($"Imported {Path.GetFileName(path)} in {clock.Elapsed.TotalMilliseconds:F1} ms:");
foreach (var layer in imported.Model)
{
    Console.WriteLine($"  {layer}");
}

foreach (var note in imported.Notes)
{
    Console.WriteLine($"  note: {note}");
}

foreach (var (key, value) in imported.Metadata)
{
    Console.WriteLine($"  metadata {key} = {value}");
}

string expectedPath = path + ".expected.json";
if (!File.Exists(expectedPath))
{
    Console.WriteLine($"\nNo {Path.GetFileName(expectedPath)}: nothing to compare with.");
    return 0;
}

var expected = JsonNode.Parse(File.ReadAllText(expectedPath))!;
var inputs = expected["inputs"]!.AsArray().Select(row => row!.AsArray().Select(v => (float)v!).ToArray()).ToList();
var reference = expected["outputs"]!.AsArray().Select(row => row!.AsArray().Select(v => (float)v!).ToArray()).ToList();
string source = expected["torch"] is { } version ? $"PyTorch {version}" : "other framework";
Console.WriteLine($"\nReference: {source} " +
    $"(trained on {expected["trainedOn"] ?? "?"}, {expected["exporter"] ?? "?"} exporter)");

float[][] Run(Func<float[,], float[,]> predict)
{
    var batch = new float[inputs.Count, inputs[0].Length];
    for (int i = 0; i < inputs.Count; i++)
    {
        for (int j = 0; j < inputs[i].Length; j++)
        {
            batch[i, j] = inputs[i][j];
        }
    }

    var result = predict(batch);
    return [.. Enumerable.Range(0, result.GetLength(0)).Select(i => Enumerable.Range(0, result.GetLength(1)).Select(j => result[i, j]).ToArray())];
}

float Compare(string name, float[][] actual)
{
    float worst = 0;
    Console.WriteLine($"\n{name}:\n  input            reference     NeuralSharp   difference");
    for (int i = 0; i < inputs.Count; i++)
    {
        for (int j = 0; j < reference[i].Length; j++)
        {
            float difference = MathF.Abs(actual[i][j] - reference[i][j]);
            worst = MathF.Max(worst, difference);
            Console.WriteLine($"  {$"[{string.Join(", ", inputs[i])}]",-16} {reference[i][j],10:F6}   {actual[i][j],10:F6}   {difference:E1}");
        }
    }

    Console.WriteLine($"  largest difference {worst:E2} ({(worst <= tolerance ? "match" : "MISMATCH")}, tolerance {tolerance:E0})");
    return worst;
}

float importedWorst = Compare($"Imported model on {device}", Run(imported.Model.Predict));

string packagePath = Path.ChangeExtension(path, ".nsm");
imported.SavePackage(packagePath);
using var predictor = Predictor.Load(packagePath, device).Build();
float packageWorst = Compare($"Saved as {Path.GetFileName(packagePath)} and reloaded on {device}",
    [.. inputs.Select(input => predictor.Predict(input))]);

clock.Restart();
const int Repeats = 1000;
for (int i = 0; i < Repeats; i++)
{
    Run(imported.Model.Predict);
}

Console.WriteLine($"\n{Repeats} batches of {inputs.Count}: {clock.Elapsed.TotalMilliseconds / Repeats:F3} ms per batch on {device}");
return importedWorst <= tolerance && packageWorst <= tolerance ? 0 : 1;
