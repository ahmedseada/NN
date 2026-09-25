# NeuralSharp

A self-contained neural network library for **.NET 10**, written in pure C#. There is **no TensorFlow.dll,
no NuGet dependencies and no native libraries of its own**. It includes tensors, automatic
differentiation, layers, losses, optimizers, a training loop, data loading, resource limits and
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
  Tensor.cs, Tensor.Operations.cs   tensors, operators, autograd
  Device.cs, ComputeResources.cs    devices; thread and memory budgets
  Autograd.cs, TensorScope.cs       NoGrad(); deterministic disposal
  Losses.cs                         MeanSquaredError, MeanAbsoluteError
  Layers/                           Module, Sequential, Linear, ReLU, Tanh, Sigmoid, Dropout
  Optimizers/                       Sgd (momentum), Adam
  Data/                             Dataset (CSV loading), StandardScaler, MinMaxScaler, DataLoader
  Training/                         Trainer, Metric, RegressionReport
  Diagnostics/                      Telemetry hub, events, ConsoleLogger, MetricsRecorder,
                                    ChannelTelemetry, JsonLinesLogger
  Backends/Cpu, Backends/Cuda       device implementations
samples/
  NeuralSharp.Samples.Xor           the classic XOR problem
  NeuralSharp.Samples.HousePrices   regression: predict house prices from a CSV file
  Shared/SampleOptions.cs           command-line options shared by the samples
tests/NeuralSharp.Tests             self-contained test runner (runs on every available device)
```

## Samples

Both samples take the same options:

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

There are 24 tests. They cover reference comparisons for every kernel, finite-difference gradient
checks for every op, optimizers, CSV parsing and scalers, data-loader coverage (including the prefetch
path), end-to-end training, telemetry subscribe/unsubscribe, memory limits, thread budgets, and
inference memory. The suite runs on every available device.

## Status

* **Verified on real hardware.** The full test suite (24 tests) passes on both the CPU and an NVIDIA
  GeForce RTX 5050 Laptop GPU (Blackwell), for 48 of 48 passing. The house-price sample gives the
  same results on both devices (R² 0.975 on CPU, 0.975 on GPU).
* The generated PTX also assembles without errors or register spills for sm_50, sm_75, sm_86 and
  sm_90 (checked with NVIDIA's `ptxas`), covering Maxwell through Hopper.
* The supported set is intentionally small: float32, 2-D matrix multiply, and the layers, losses and
  activations listed above.
* Small models such as the samples run at about the same speed on CPU and GPU, because each step is
  dominated by kernel-launch overhead. The GPU pays off with wide layers and large batches.
