using System.Collections;

namespace NN.Layers;

/// <summary>
/// Runs modules one after another, feeding each output into the next.
/// Supports collection-initializer syntax:
/// <code>
/// var model = new Sequential
/// {
///     new Linear(2, 8),
///     new Tanh(),
///     new Linear(8, 1),
///     new Sigmoid(),
/// };
/// </code>
/// </summary>
public sealed class Sequential : Module, IEnumerable<Module>
{
    private readonly List<Module> _modules = [];

    /// <summary>Creates an empty container; add layers with <see cref="Add"/> or a collection initializer.</summary>
    public Sequential()
    {
    }

    /// <summary>Creates a container from the given layers.</summary>
    public Sequential(params IEnumerable<Module> modules) => _modules.AddRange(modules);

    /// <summary>The number of layers.</summary>
    public int Count => _modules.Count;

    /// <summary>The layer at <paramref name="index"/>.</summary>
    public Module this[int index] => _modules[index];

    /// <summary>Appends a layer.</summary>
    public void Add(Module module) => _modules.Add(module ?? throw new ArgumentNullException(nameof(module)));

    /// <inheritdoc />
    public override Tensor Forward(Tensor input)
    {
        var x = input;
        foreach (var module in _modules)
        {
            x = module.Forward(x);
        }

        return x;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => _modules.SelectMany(m => m.Parameters());

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        foreach (var module in _modules)
        {
            module.MoveTo(device);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var module in _modules)
        {
            module.Dispose();
        }

        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public IEnumerator<Module> GetEnumerator() => _modules.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public override string ToString() => "Sequential(\n" + string.Join("\n", _modules.Select((m, i) => $"  ({i}) {m}")) + "\n)";
}
