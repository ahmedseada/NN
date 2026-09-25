# NeuralSharp

A self-contained neural network library for **.NET 10**, written in pure C#. There is **no TensorFlow.dll,
no NuGet dependencies and no native libraries of its own**. It includes N-D tensors with automatic
differentiation; dense, convolutional, recurrent and transformer layers; regression and classification
losses; optimizers with learning-rate schedules; a training loop; data loading; resource limits; and
telemetry hooks.

| Backend | How it works | Requirements |
|---------|--------------|--------------|
| **CPU** | `Vector<T>` SIMD kernels (AVX2/AVX-512/NEON), a register-blocked matrix multiply, multi-threading for large tensors, pooled buffers | Any machine running .NET 10 |
| **CUDA** | P/Invoke straight into the NVIDIA **driver** API (`nvcuda.dll` / `libcuda.so.1`). The GPU kernels are PTX assembly generated in C# (`PtxKernels.cs`), and the driver JIT-compiles them for the installed GPU | An NVIDIA GPU and display driver. No CUDA Toolkit, cuBLAS, cuDNN or NVRTC |

`Device.Default` picks the first GPU when a driver is present and falls back to the CPU otherwise.
Set `NEURALSHARP_DISABLE_CUDA=1` to force the CPU.

## Layout

```
src/NeuralSharp/
  Tensor*.cs                        N-D tensors, operators, autograd, shape ops, number-type interop
  Device.cs, ComputeResources.cs    devices; thread and memory budgets
  Autograd.cs, TensorScope.cs       NoGrad(); deterministic disposal
  Losses.cs                         MSE, MAE, CrossEntropy, BinaryCrossEntropy(WithLogits)
  Layers/                           Linear, Conv2d, MaxPool2d, GlobalAveragePool2d, Flatten,
                                    BatchNorm, LayerNorm, Embedding, LSTM, GRU,
                                    MultiHeadAttention, TransformerEncoderLayer, PositionalEncoding,
                                    ReLU, Tanh, Sigmoid, GELU, Softmax, Dropout, Lambda, Sequential
  Optimizers/                       Sgd, Adam, AdamW, schedulers (step, exponential, cosine + warm-up)
  Data/                             Dataset (CSV, class labels, feature shapes), scalers, DataLoader
  Training/                         Trainer, Metric (MAE, RMSE, Accuracy), RegressionReport
  Diagnostics/                      Telemetry hub, events, ConsoleLogger, MetricsRecorder,
                                    ChannelTelemetry, JsonLinesLogger
  Backends/Cpu, Backends/Cuda       device implementations (CPU SIMD kernels, PTX kernels)
samples/
  NeuralSharp.Samples.Xor             the classic XOR problem
  NeuralSharp.Samples.HousePrices     regression: predict house prices from a CSV file
  NeuralSharp.Samples.Classification  multi-class: 3 spirals, softmax + cross-entropy, BatchNorm
  NeuralSharp.Samples.Images          CNN: classify drawn shapes (Conv2d, MaxPool2d, BatchNorm)
  NeuralSharp.Samples.Sequences       sentiment with negation: bag-of-words vs LSTM, GRU, Transformer
  Shared/SampleOptions.cs             command-line options shared by the samples
tests/NeuralSharp.Tests             self-contained test runner (runs on every available device)
```

## Samples

| Sample | Shows | Typical CPU result |
|--------|-------|--------------------|
| `Xor` | smallest possible network | 4/4 correct in 0.1 s |
| `HousePrices` | CSV loading, scaling, regression, early stopping | R² 0.975, 5.3% mean error, 1 s |
| `Classification` | softmax + cross-entropy, BatchNorm, AdamW, cosine schedule, confusion matrix | 97.8% accuracy, 2.4 s |
| `Images` | Conv2d, MaxPool2d, BatchNorm, Flatten on 16×16 images | 100% accuracy, about 11 s |
| `Sequences` | Embedding, LSTM, GRU, Transformer vs an order-blind baseline | LSTM/GRU/Transformer 99.5–100%, bag of words 70% |

All samples take the same options:

```bash
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices               # GPU if available, else CPU
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cpu      # force CPU
dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices -- --cuda     # force GPU
dotnet run -c Release --project samples/NeuralSharp.Samples.Xor -- --help
```

| Option | Meaning |
|--------|---------|
| `--device auto\|cpu\|cuda\|cuda:N`, `--cpu`, `--cuda` | where to run |
| `--threads N` | CPU threads to use (default: all cores) |
| `--gpu-memory MiB`, `--cpu-memory MiB` | cap tensor memory (default: unlimited) |
| `--epochs N`, `--batch-size N` | training settings |
| `--log training,batches,gradients,layers,operations,inference,all` | what the console logger prints |
| `--log-file run.jsonl` | also write telemetry as JSON Lines |
| `--data path.csv` | dataset file (house prices) |

The house-price sample loads `data/houses.csv`: 2,500 synthetic but realistic listings with 9 features
(area, bedrooms, bathrooms, age, distance to the city centre, quality, garage, pool, lot size).
It splits them 80/20, standardizes features and prices, and trains a 9→64→32→1 network with
dropout, Adam and early stopping. It then reports test error in dollars, prices three new listings,
and saves the weights, scalers and a training-history CSV. A typical CPU run reaches **R² ≈ 0.97 and
about 5% mean error** in about a second, close to the ±6% noise built into the data.

## Using the library

### Training with the `Trainer`

```csharp
using NeuralSharp;
using NeuralSharp.Data;
using NeuralSharp.Layers;
using NeuralSharp.Optimizers;
using NeuralSharp.Training;

var data = Dataset.LoadCsv("houses.csv", new CsvOptions { TargetColumns = ["price"], IgnoreColumns = ["id"] });
var (train, test) = data.Split(0.8, seed: 1);

var x = StandardScaler.FitFeatures(train);
var y = StandardScaler.FitTargets(train);
var trainLoader = new DataLoader(train.Scale(x, y), batchSize: 64, shuffle: true);
var testLoader  = new DataLoader(test.Scale(x, y), batchSize: 512);

using var model = new Sequential
{
    new Linear(data.FeatureCount, 64), new ReLU(), new Dropout(0.05f),
    new Linear(64, 32), new ReLU(),
    new Linear(32, 1),
};

var trainer = new Trainer(model, new Adam(model.Parameters(), 2e-3f), Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
    EarlyStoppingPatience = 30,             // restores the best weights when it stops
};
TrainingHistory history = trainer.Fit(trainLoader, epochs: 400, validation: testLoader);
float[,] predictions = trainer.Predict(test.Scale(x, y));
```

### Low-level: tensors and a hand-written loop

```csharp
var a = Tensor.From(new float[,] { { 1, 2 }, { 3, 4 } }, requiresGrad: true);
var loss = (a.MatMul(a).Relu() * 2f + 1f).Mean();
loss.Backward();                      // a.Grad holds d(loss)/da

for (int epoch = 0; epoch < 1000; epoch++)
{
    using var scope = new TensorScope();   // releases this iteration's tensors
    var l = Losses.MeanSquaredError(model.Forward(inputs), targets);
    optimizer.ZeroGrad();
    l.Backward();
    optimizer.Step();
}

using var prediction = model.Predict(inputs);   // evaluation mode, no gradients
```

### Classification

Classifiers output raw scores (logits). `CrossEntropy` applies log-softmax itself, which is more
stable than a separate softmax layer. Add `Softmax` only when you want probabilities at inference time.

```csharp
var data = Dataset.FromClassLabels(features, labels, classes: 3);      // one-hot targets
// or: Dataset.LoadCsv(path, new CsvOptions { TargetColumns = ["label"] }).ToOneHot(3)

var trainer = new Trainer(model, new AdamW(model.Parameters(), 1e-3f), (p, t) => Losses.CrossEntropy(p, t))
{
    Metrics = { Metric.Accuracy },
    Scheduler = new CosineAnnealing(optimizer, totalEpochs: 100, warmupEpochs: 5),
};
using var probabilities = model.Predict(x).Softmax();
using var classes = probabilities.ArgMax();
```

For two classes with one output column, use `Losses.BinaryCrossEntropyWithLogits` and
`Metric.BinaryAccuracy(threshold: 0)`.

### Images (CNN)

```csharp
var images = Dataset.FromClassLabels(pixels, labels, 10).WithFeatureShape(1, 28, 28);   // batches are [N, 1, 28, 28]
var cnn = new Sequential
{
    new Conv2d(1, 32, kernelSize: 3, padding: 1), new BatchNorm(32), new ReLU(), new MaxPool2d(2),
    new Conv2d(32, 64, kernelSize: 3, padding: 1), new BatchNorm(64), new ReLU(), new MaxPool2d(2),
    new Flatten(), new Linear(64 * 7 * 7, 128), new ReLU(), new Dropout(0.3f), new Linear(128, 10),
};
```

`Conv2d` unfolds image patches (im2col) and runs one large matrix product on the same optimized GEMM
as `Linear`, on both CPU and GPU.

### Sequences (RNN and transformer)

Token ids go in as floats, shape [batch, time]:

```csharp
var lstm = new Sequential
{
    new Embedding(vocabulary, 64), new LSTM(64, 128), new Linear(128, classes),   // LSTM returns the last state
};

var transformer = new Sequential
{
    new Embedding(vocabulary, 64), new PositionalEncoding(maxLength, 64),
    new TransformerEncoderLayer(64, heads: 4), new TransformerEncoderLayer(64, heads: 4),
    new LayerNorm(64), new Lambda(x => x.Mean(1), "MeanOverTime"), new Linear(64, classes),
};
```

Use `LSTM(..., returnSequences: true)` or `GRU` for per-step outputs. `MultiHeadAttention(dim, heads, causal: true)`
masks future positions for autoregressive models. Setting `Trainer.MaxGradientNorm` clips gradients,
which recurrent networks usually need.

### Tensor operations

N-D tensors support batched `MatMul` (with transpose flags), `Permute`, `Transpose`, `Narrow`,
`Tensor.Concat`, `Tensor.Stack`, `Flatten`, `Sum(dim)`, `Mean(dim)`, `Softmax`, `LogSoftmax`,
`ArgMax`, `Exp`, `Log` and `Gelu`. Adding a tensor whose shape matches the trailing dimensions
broadcasts it, as with a bias [F] over [N, F] or a mask [T, T] over [B, T, T]. All of these are
differentiated automatically.

### Other number types

Every kernel computes in float32, the standard for neural networks and the only type with fast
SIMD and GPU paths everywhere. Data of any numeric type converts at the edges:

```csharp
using var t = Tensor.From<double>(doubles, [rows, cols]);   // also int, long, byte, Half, decimal, ...
using var m = Tensor.From(new int[,] { { 1, 2 }, { 3, 4 } });
int[] ints = t.ToArray<int>();                               // saturating, truncating conversion
```

### Optimizers and schedules

`Sgd(momentum, weightDecay)`, `Adam(weightDecay)` (L2) and `AdamW` (decoupled weight decay) are
available, together with the `StepDecay`, `ExponentialDecay`, `CosineAnnealing(warmupEpochs)` and
`LambdaSchedule` schedulers. `optimizer.ClipGradientNorm(max)` and `optimizer.GradientNorm()` work
in hand-written loops too.

## Telemetry: logging and tracking

Every step of training and inference is published to a hub, and you choose what to listen to.
Hooks are **zero-cost when nothing is subscribed**. Each publishing site checks one static field
with a bitwise AND, and no event is built or allocated unless some hook asked for that level.

```csharp
using NeuralSharp.Diagnostics;

using var a = Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, epochInterval: 10));
using var b = Telemetry.Subscribe(new MetricsRecorder());                        // in memory, SaveCsv()
await using var file = new JsonLinesLogger("run.jsonl", TelemetryLevel.All);     // background writer
using var c = Telemetry.Subscribe(file);
```

| Level | Event | Data |
|-------|-------|------|
| `Training` | `TrainingStarted`, `EpochCompleted`, `TrainingCompleted` | model summary, device, sizes; per-epoch loss, metrics, validation loss/metrics, learning rate, duration, samples/s, memory, best flag; early-stop info |
| `Batches` | `BatchCompleted` | epoch, batch, global step, batch loss, learning rate, data-wait time, compute time |
| `Gradients` | (adds to `BatchCompleted`) | global gradient L2 norm |
| `Layers` | `LayerForward` | layer name/type, nesting depth, input and output shapes, time, train/eval mode |
| `Operations` | `OperationCompleted` | every tensor op, forward and backward (∇), shape, device, time |
| `Inference` | `InferenceCompleted` | model, samples, shapes, device, latency, samples/s |

**Custom hooks.** Implement `ITelemetryHook`, set `Levels`, and override only the methods you need.
The other methods default to no-ops. Events are `readonly record struct`s passed by `in`, so they
are not copied or boxed.

```csharp
sealed class SlackAlert : ITelemetryHook
{
    public TelemetryLevel Levels => TelemetryLevel.Training;
    public void OnEpochCompleted(in EpochCompleted e) { if (double.IsNaN(e.Loss)) Alert("loss diverged"); }
}
```

**Why synchronous hooks plus an optional `Channel`.** A direct interface call is the fastest possible
dispatch, with no queueing, allocation or thread hop. Hooks that do slow work, such as files,
databases or HTTP, should use `ChannelTelemetry`. It copies each event into a bounded `Channel`
that never blocks training (the oldest events are dropped when it is full), and your consumer reads
it with `await foreach`. `JsonLinesLogger` is built this way.

GPU work is asynchronous, so layer and operation timings measure launch time by default. Set
`Telemetry.SynchronizeForTiming = true` when profiling to get true execution times.

## Data loading

* `Dataset.LoadCsv(path, options)` reads numeric CSV files. Lines are parsed in parallel, and errors
  name the line and column. `Dataset.FromArrays` and `Dataset.FromFlat` build a dataset from memory.
* `Split`, `Subset` and `Scale` return new, immutable datasets.
* `StandardScaler` and `MinMaxScaler` are fitted on training data. They provide `Transform` and
  `InverseTransform` (to turn predictions back into real units), plus `Save`/`Load` for deployment.
* `DataLoader` batches and shuffles data and moves each batch to the device. For large batches, the
  next batch is gathered on a worker thread while the current one trains, using double-buffered
  prefetch.

## Resources and parallelism

```csharp
ComputeResources.MaxCpuThreads  = 4;                   // default: every core
ComputeResources.GpuMemoryLimit = 2L << 30;            // bytes per GPU; default: unlimited
ComputeResources.CpuMemoryLimit = 1L << 30;            // default: unlimited
MemoryUsage usage = ComputeResources.GetMemoryUsage(Device.Default);   // in use / cached / limit
ComputeResources.ReleaseCachedMemory();
```

By default nothing is reserved up front, and memory is allocated as the network needs it. Freed blocks
are cached and reused, so a training loop stops allocating after its first iteration. Going over a
limit first releases the cache, then throws `ResourceLimitExceededException` with a clear message.

Parallelism is used where it pays off:

* SIMD in every CPU kernel.
* Multi-threaded kernels above 65,536 elements, and a multi-threaded matrix multiply for large
  products. On 4 AVX2 cores, a 1024×1024 multiply reaches about 86 GFLOP/s.
* Parallel CSV parsing.
* Background batch prefetch.
* On the GPU, thousands of threads per operation. Loss and metric sums stay on the device, so each
  epoch needs only one synchronization.

Everything on the CPU respects `MaxCpuThreads`, and small tensors stay single-threaded because
scheduling would cost more than it saves.

## NPU support

NPUs (Intel AI Boost, AMD Ryzen AI / XDNA, Qualcomm Hexagon) are **not supported**, and they cannot be
supported under this library's no-dependency rule:

* Unlike NVIDIA GPUs, which accept PTX through the driver, NPUs expose no general-purpose instruction
  set. They only run networks compiled by the vendor's own toolchain: OpenVINO for Intel, Ryzen AI /
  Vitis AI for AMD, QNN for Qualcomm, or DirectML / Windows ML on Windows.
* Those toolchains are the "prebuilt libraries" the project excludes.
* Current NPUs are mostly inference engines (INT8/FP16) and do not support training.

The backend design (`Backends/Backend.cs`) leaves room for an optional add-on package, for example
`NeuralSharp.OpenVino`, that runs inference on an NPU while the core stays dependency-free.

## Tests

```bash
dotnet run -c Release --project tests/NeuralSharp.Tests
```

There are 50 tests. They cover reference comparisons for every kernel (matrix products, softmax,
convolution and pooling against direct implementations) and finite-difference gradient checks for
every op and layer, including their weights. They also cover end-to-end learning (regression, spiral
classification, a CNN, LSTM and transformer sequence models), optimizers and schedules, CSV parsing,
data loading, telemetry, memory limits and thread budgets. The suite runs on every available device.

## Status

* **Verified on real hardware (main branch).** The original 24 tests pass on both the CPU and an
  NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell).
* **This branch** adds 26 tests (50 in total) covering classification, normalization, embeddings,
  convolution, pooling, LSTM/GRU, attention, schedulers and number types. They pass on the CPU.
  The 26 new GPU kernels (51 in total) assemble without errors or register spills for sm_50, sm_75,
  sm_86, sm_90 and sm_120 (checked with `ptxas`), but they still need a run on a GPU. Run
  `dotnet run -c Release --project tests/NeuralSharp.Tests` on an NVIDIA machine to verify them.
* Everything computes in float32. Tensors hold up to 2³¹ elements, and embedding ids must be below
  2²⁴ (the largest integer a float stores exactly).
