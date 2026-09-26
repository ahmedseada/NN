using System.Collections;

namespace NeuralSharp.Layers;

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
public sealed class Sequential : Module, IEnumerable<Module>, ICachedModule
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
    protected override Tensor ForwardCore(Tensor input)
    {
        var x = input;
        foreach (var module in _modules)
        {
            x = module.Forward(x);
        }

        return x;
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _modules;

    /// <summary>
    /// Incremental forward pass: layers implementing <see cref="ICachedModule"/> use their cached path, others run
    /// normally on the new positions only. Brackets the pass with <see cref="DecodingContext.BeginStep"/> /
    /// <see cref="DecodingContext.EndStep"/> when called at the top level.
    /// </summary>
    public Tensor ForwardCached(Tensor input, DecodingContext context)
    {
        bool outermost = !context.InStep;
        int steps = input.Shape[1];
        if (outermost)
        {
            context.BeginStep(steps);
        }

        // With LastPositionOnly, the layers after the last cached one see only the last position.
        int lastCached = outermost && context.LastPositionOnly && steps > 1 ? _modules.FindLastIndex(m => m is ICachedModule) : -1;
        var x = input;
        for (int i = 0; i < _modules.Count; i++)
        {
            var module = _modules[i];
            x = module is ICachedModule cached ? cached.ForwardCached(x, context) : module.Forward(x);
            if (i == lastCached && x.Rank == 3)
            {
                x = x.Narrow(1, x.Shape[1] - 1, 1);
            }
        }

        if (outermost)
        {
            context.EndStep(steps);
        }

        return x;
    }

    /// <inheritdoc />
    public IEnumerator<Module> GetEnumerator() => _modules.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public override string ToString() => $"Sequential({_modules.Count} layers)";
}
