# NN — a self-contained neural network library for .NET 10

A TensorFlow/PyTorch-style library written in pure C#. There is **no TensorFlow.dll and no NuGet
dependencies**, and no native libraries of its own. It has tensors, automatic differentiation, layers,
losses and optimizers, and runs on two backends:

| Backend | How it works | Requirements |
|---------|--------------|--------------|
| **CPU** | `Vector<T>` SIMD kernels (AVX2/AVX-512/NEON), a register-blocked matrix multiply, `Parallel.For` for large tensors, pooled buffers | Any machine running .NET 10 |
| **CUDA** | P/Invoke straight into the NVIDIA **driver** API (`nvcuda.dll` / `libcuda.so.1`). The GPU kernels are PTX assembly generated in C# (`PtxKernels.cs`), and the driver JIT-compiles them for the installed GPU | An NVIDIA GPU and display driver. No CUDA Toolkit, cuBLAS, cuDNN or NVRTC |

`Device.Default` picks the first GPU when a driver is present and falls back to the CPU otherwise.
Set `NN_DISABLE_CUDA=1` to force the CPU.

## Layout

```
src/NN/                 the library
  Tensor.cs             tensor type, creation, data access, autograd engine
  Tensor.Operations.cs  operators (+ - * /), MatMul, activations, reductions and their gradients
  Autograd.cs           Autograd.NoGrad()
  TensorScope.cs        deterministic disposal of intermediates in training loops
  Losses.cs             MeanSquaredError
  Layers/               Module, Linear, Sequential, Sigmoid, Tanh, ReLU
  Optimizers/           Sgd (with momentum), Adam
  Backends/Cpu/         SIMD + multi-threaded CPU kernels, GEMM
  Backends/Cuda/        driver bindings, caching GPU allocator, PTX kernel generator
samples/NN.Xor/         console app that learns XOR
tests/NN.Tests/         self-contained test runner (gradient checks, reference comparisons)
```

## Quick start: XOR

```csharp
using NN;
using NN.Layers;
using NN.Optimizers;

using var inputs  = Tensor.From(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
using var targets = Tensor.From(new float[,] { { 0 }, { 1 }, { 1 }, { 0 } });

using var model = new Sequential
{
    new Linear(2, 8),
    new Tanh(),
    new Linear(8, 1),
    new Sigmoid(),
};
using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);

for (int epoch = 0; epoch < 2000; epoch++)
{
    using var scope = new TensorScope();          // frees this iteration's tensors
    var loss = Losses.MeanSquaredError(model.Forward(inputs), targets);
    optimizer.ZeroGrad();
    loss.Backward();
    optimizer.Step();
}

using (Autograd.NoGrad())
    Console.WriteLine(model.Forward(inputs));
```

Run the sample:

```bash
dotnet run -c Release --project samples/NN.Xor            # GPU if available
dotnet run -c Release --project samples/NN.Xor -- --cpu   # force CPU
dotnet run -c Release --project samples/NN.Xor -- --cuda  # force GPU
```

Run the tests (every test runs on the CPU and, when present, on the GPU):

```bash
dotnet run -c Release --project tests/NN.Tests
```

## Working with tensors

```csharp
var a = Tensor.From(new float[,] { { 1, 2 }, { 3, 4 } }, requiresGrad: true);
var b = Tensor.Uniform([2, 3], -1f, 1f);        // collection-expression shapes
var c = (a.MatMul(b).Relu() * 2f + 1f).Mean();
c.Backward();                                    // a.Grad now holds dc/da
float value = c.Item();
float[,] data = a.ToArray2D();
var onGpu = a.To(Device.Cuda());
```

* `+ - *` between tensors are element-wise. A 1-D right operand is broadcast over rows, which is how biases are added.
* `MatMul` multiplies [m, k] by [k, n].
* Tensors own device memory. Dispose them, or create them inside a `TensorScope`. Parameters,
  gradients and optimizer state are never captured by a scope. A finalizer is the safety net.
* Save and restore weights with `model.Save(path)` and `model.Load(path)`.

## Performance notes

* Float32 everywhere, with contiguous row-major storage.
* CPU GEMM keeps a 4 × (2·SIMD width) tile of C in registers. On a 4-core AVX2 machine it reaches
  about 86 GFLOP/s for 1024×1024 matrices.
* Element-wise kernels are struct-based. There are no delegates or allocations per call, and work
  goes multi-threaded above 65,536 elements.
* Both backends pool memory by size, so a steady training loop stops allocating after its first iteration.
* On the GPU, data stays resident. Copies happen only on `ToArray`/`Item`/`To`. Matrix multiply
  uses 16×16 shared-memory tiles, and sums use a block tree reduction plus atomics.
* Tiny models such as XOR are dominated by launch overhead, so they run faster on the CPU. The GPU
  pays off with large batches and layers.

## Status

* The CPU backend is covered by `tests/NN.Tests`.
* The generated PTX assembles without errors or register spills for sm_50, sm_75, sm_86 and sm_90
  (checked with NVIDIA's `ptxas`). It has **not yet been executed on real GPU hardware**. Running
  `tests/NN.Tests` on a machine with an NVIDIA GPU runs the whole suite against CUDA as well.
* The supported set is intentionally small: one dtype (float32), 2-D matrix multiply, row-broadcast
  add, and the activations and losses listed above.
