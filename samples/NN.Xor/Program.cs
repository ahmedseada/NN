// Trains a small neural network to learn XOR, the classic problem a single linear layer cannot solve.
//
//   dotnet run -c Release --project samples/NN.Xor            GPU if available, otherwise CPU
//   dotnet run -c Release --project samples/NN.Xor -- --cpu   force the CPU
//   dotnet run -c Release --project samples/NN.Xor -- --cuda  force the GPU

using System.Diagnostics;
using NN;
using NN.Layers;
using NN.Optimizers;

var device = args.Contains("--cpu") ? Device.Cpu
    : args.Contains("--cuda") ? Device.Cuda()
    : Device.Default;
Device.Default = device;

Console.WriteLine($"Device: {device} - {device.Name}");

// The four XOR cases: output is 1 when exactly one input is 1.
using var inputs = Tensor.From(new float[,]
{
    { 0, 0 },
    { 0, 1 },
    { 1, 0 },
    { 1, 1 },
});

using var targets = Tensor.From(new float[,]
{
    { 0 },
    { 1 },
    { 1 },
    { 0 },
});

// 2 inputs -> 8 hidden units (tanh) -> 1 output (sigmoid, so it reads as a probability).
var random = new Random(42);
using var model = new Sequential
{
    new Linear(2, 8, random: random),
    new Tanh(),
    new Linear(8, 1, random: random),
    new Sigmoid(),
};

using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);

Console.WriteLine(model);
Console.WriteLine($"Trainable parameters: {model.ParameterCount}");
Console.WriteLine();

const int Epochs = 2000;
var stopwatch = Stopwatch.StartNew();

for (int epoch = 1; epoch <= Epochs; epoch++)
{
    // Everything created inside the scope (activations, the loss) is released at the end of the iteration.
    using var scope = new TensorScope();

    var predictions = model.Forward(inputs);
    var loss = Losses.MeanSquaredError(predictions, targets);

    optimizer.ZeroGrad();
    loss.Backward();
    optimizer.Step();

    if (epoch == 1 || epoch % 200 == 0)
    {
        Console.WriteLine($"Epoch {epoch,5}  loss {loss.Item():F6}");
    }
}

device.Synchronize();
stopwatch.Stop();
Console.WriteLine($"\nTrained {Epochs} epochs in {stopwatch.Elapsed.TotalMilliseconds:F0} ms " +
                  $"({stopwatch.Elapsed.TotalMicroseconds / Epochs:F1} µs per epoch)\n");

// Inference: no gradient bookkeeping needed.
float[,] results;
using (Autograd.NoGrad())
using (var output = model.Forward(inputs))
{
    results = output.ToArray2D();
}

var x = inputs.ToArray2D();
var y = targets.ToArray2D();
int correct = 0;

Console.WriteLine(" A  B | expected | predicted | rounded");
Console.WriteLine("------+----------+-----------+--------");
for (int i = 0; i < 4; i++)
{
    int rounded = results[i, 0] >= 0.5f ? 1 : 0;
    bool ok = rounded == (int)y[i, 0];
    correct += ok ? 1 : 0;
    Console.WriteLine($" {x[i, 0]}  {x[i, 1]} |    {y[i, 0]}     |  {results[i, 0]:F4}   |   {rounded} {(ok ? "✓" : "✗")}");
}

Console.WriteLine($"\n{correct}/4 correct");
return correct == 4 ? 0 : 1;
