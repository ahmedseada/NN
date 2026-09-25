// Predicts house prices from 9 features with a small neural network: load CSV -> split -> scale ->
// train with validation and early stopping -> evaluate in dollars -> predict new houses -> save.
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices               GPU if available, else CPU
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cpu
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cuda
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --threads 2 --log training,inference --log-file run.jsonl

using System.Diagnostics;
using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Diagnostics;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;
using NeuralSharp.Samples;
using NeuralSharp.Samples.HousePrices;
using NeuralSharp.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}");
Console.WriteLine($"CPU threads: {ComputeResources.MaxCpuThreads}\n");

// ---------------------------------------------------------------- telemetry
// Console: every 10th epoch (plus the first and last). Recorder: full history for a CSV report.
// Optional JSON Lines file for dashboards / later analysis, written on a background thread.
var recorder = new MetricsRecorder();
using var consoleHook = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 10, batchInterval: 10));
using var recorderHook = Telemetry.Subscribe(recorder);
await using var fileLog = options.LogFile is { } logPath ? new JsonLinesLogger(logPath, TelemetryLevel.All & ~TelemetryLevel.Operations) : null;
using var fileHook = fileLog is null ? null : Telemetry.Subscribe(fileLog);

// ---------------------------------------------------------------- data
string dataPath = options.DataFile ?? Path.Combine(AppContext.BaseDirectory, "data", "houses.csv");
if (!File.Exists(dataPath))
{
    Console.WriteLine($"{dataPath} not found; generating a synthetic dataset.");
    HouseDataGenerator.Write(dataPath);
}

var stopwatch = Stopwatch.StartNew();
var houses = Dataset.LoadCsv(dataPath, new CsvOptions
{
    TargetColumns = ["price"],
    IgnoreColumns = ["id"],
});
Console.WriteLine($"Loaded {houses} in {stopwatch.ElapsedMilliseconds} ms");

var (train, test) = houses.Split(trainFraction: 0.8, seed: 1);
Console.WriteLine($"Split: {train.Count} training / {test.Count} test samples\n");

// Networks train best on inputs and outputs around zero with unit scale. Fit the scalers on the
// training set only, so no information from the test set leaks into training.
var featureScaler = StandardScaler.FitFeatures(train);
var priceScaler = StandardScaler.FitTargets(train);
var trainScaled = train.Scale(featureScaler, priceScaler);
var testScaled = test.Scale(featureScaler, priceScaler);

int batchSize = options.BatchSize ?? 64;
var trainLoader = new DataLoader(trainScaled, batchSize, shuffle: true, device: device, seed: 1);
var testLoader = new DataLoader(testScaled, batchSize: 512, device: device);

// ---------------------------------------------------------------- model
var random = new Random(1);
using var model = new Sequential
{
    new Linear(houses.FeatureCount, 64, device: device, random: random),
    new ReLU(),
    new Dropout(0.05f, random),
    new Linear(64, 32, device: device, random: random),
    new ReLU(),
    new Linear(32, 1, device: device, random: random),
};
model.Name = "house-price-mlp";

using var optimizer = new Adam(model.Parameters(), learningRate: 0.002f);
var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    EarlyStoppingPatience = 30,
};

// ---------------------------------------------------------------- train
var history = trainer.Fit(trainLoader, epochs: options.Epochs ?? 400, validation: testLoader);
Console.WriteLine($"Best epoch: {history.BestEpoch} (validation loss {history.BestLoss:F5}){(history.StoppedEarly ? ", stopped early and restored its weights" : "")}\n");

// ---------------------------------------------------------------- evaluate in dollars
var scaledPredictions = trainer.Predict(testScaled);
float[] predicted = [.. scaledPredictions.Cast<float>()];
priceScaler.InverseTransform(predicted, 1);
var report = RegressionReport.Compute(predicted, test.Targets);

Console.WriteLine("Test set:");
Console.WriteLine($"  Mean absolute error:     ${report.MeanAbsoluteError:N0}");
Console.WriteLine($"  Root mean squared error: ${report.RootMeanSquaredError:N0}");
Console.WriteLine($"  Mean absolute % error:   {report.MeanAbsolutePercentageError:P1}");
Console.WriteLine($"  R²:                      {report.RSquared:F4}\n");

Console.WriteLine("   area  beds baths  age  dist qual |     actual |  predicted |  error");
Console.WriteLine("-----------------------------------+------------+------------+-------");
for (int i = 0; i < 8; i++)
{
    var f = test.GetFeatures(i);
    float actual = test.GetTargets(i)[0];
    Console.WriteLine($"  {f[0],5:F0}  {f[1],4:F0} {f[2],5:F0} {f[3],4:F0} {f[4],5:F1} {f[5],4:F0} | {Money(actual),10} | {Money(predicted[i]),10} | {(predicted[i] - actual) / actual,6:P1}");
}

// ---------------------------------------------------------------- predict new houses
// Same column order as the CSV features: area, bedrooms, bathrooms, age, distance, quality, garage, pool, lot.
string[] descriptions =
[
    "Starter home, far from the city",
    "Family house in the suburbs",
    "New luxury villa close to downtown",
];
float[,] newHouses =
{
    { 950, 2, 1, 45, 30.0f, 4, 0, 0, 2_900 },
    { 2_100, 4, 2, 15, 9.5f, 7, 2, 0, 6_500 },
    { 4_200, 5, 4, 2, 2.0f, 10, 3, 1, 12_000 },
};

float[] scaledInput = [.. newHouses.Cast<float>()];
featureScaler.Transform(scaledInput, houses.FeatureCount);
using var inputTensor = Tensor.From(scaledInput, [descriptions.Length, houses.FeatureCount], device);
using var outputTensor = model.Predict(inputTensor); // publishes an InferenceCompleted event when inference logging is on
float[] prices = outputTensor.ToArray();
priceScaler.InverseTransform(prices, 1);

Console.WriteLine("\nPredictions for new listings:");
for (int i = 0; i < descriptions.Length; i++)
{
    Console.WriteLine($"  {descriptions[i],-36} {Money(prices[i]),12}");
}

// ---------------------------------------------------------------- save
string outputDir = Path.Combine(AppContext.BaseDirectory, "output");
Directory.CreateDirectory(outputDir);
model.Save(Path.Combine(outputDir, "house-price.weights"));
featureScaler.Save(Path.Combine(outputDir, "feature-scaler.txt"));
priceScaler.Save(Path.Combine(outputDir, "price-scaler.txt"));
recorder.SaveCsv(Path.Combine(outputDir, "training-history.csv"));
Console.WriteLine($"\nSaved model, scalers and training history to {outputDir}");
Console.WriteLine($"Memory on {device}: {ComputeResources.GetMemoryUsage(device)}");

return report.RSquared > 0.8 ? 0 : 1;

static string Money(float value) => "$" + value.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);
