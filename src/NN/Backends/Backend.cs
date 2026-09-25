namespace NN.Backends;

internal enum UnaryOp
{
    Sigmoid,
    Tanh,
    Relu,
    Square,
}

internal enum BinaryOp
{
    Add,
    Sub,
    Mul,
}

/// <summary>
/// A reference-counted block of device memory holding <see cref="Length"/> floats.
/// When the last reference is released the block goes back to its backend's pool.
/// </summary>
internal abstract class Storage(Backend backend, int length)
{
    private int _refs = 1;

    public Backend Backend { get; } = backend;

    public int Length { get; } = length;

    public void AddRef() => Interlocked.Increment(ref _refs);

    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0)
        {
            Backend.Return(this);
        }
    }
}

/// <summary>
/// The math primitives a device must provide. Every operation works on contiguous
/// row-major float32 buffers. Methods named "...Backward", <see cref="Axpy"/>,
/// <see cref="MulAdd"/>, <see cref="SumRows"/> and <see cref="AddBroadcastScalar"/>
/// accumulate into their output (+=) so gradients from several paths add up.
/// </summary>
internal abstract class Backend
{
    public abstract Storage Allocate(int length, bool zeroed);

    public abstract void Return(Storage storage);

    public abstract void Upload(ReadOnlySpan<float> source, Storage destination);

    public abstract void Download(Storage source, Span<float> destination);

    public abstract void Fill(Storage y, int n, float value);

    public abstract void Copy(Storage x, Storage y, int n);

    /// <summary>y = op(x).</summary>
    public abstract void Unary(UnaryOp op, Storage x, Storage y, int n);

    /// <summary>dx += dy * op'(x), where y = op(x) is passed in for ops whose derivative is cheaper from y.</summary>
    public abstract void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n);

    /// <summary>c = a op b, element-wise.</summary>
    public abstract void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n);

    /// <summary>y = alpha * x + beta.</summary>
    public abstract void Affine(Storage x, Storage y, int n, float alpha, float beta);

    /// <summary>y += alpha * x.</summary>
    public abstract void Axpy(Storage x, Storage y, int n, float alpha);

    /// <summary>c += a * b, element-wise.</summary>
    public abstract void MulAdd(Storage a, Storage b, Storage c, int n);

    /// <summary>c[r, j] = a[r, j] + v[j].</summary>
    public abstract void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols);

    /// <summary>y[j] += sum over r of x[r, j].</summary>
    public abstract void SumRows(Storage x, Storage y, int rows, int cols);

    /// <summary>result[0] = scale * sum(x).</summary>
    public abstract void Sum(Storage x, Storage result, int n, float scale);

    /// <summary>y[i] += scale * s[0].</summary>
    public abstract void AddBroadcastScalar(Storage s, Storage y, int n, float scale);

    /// <summary>
    /// c[m, n] = op(a) * op(b) + beta * c, where op(a) is [m, k] and op(b) is [k, n].
    /// With <paramref name="transA"/> a is stored as [k, m]; with <paramref name="transB"/> b is stored as [n, k].
    /// </summary>
    public abstract void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta);

    /// <summary>SGD with optional momentum: v = momentum * v + g; p -= lr * v (v is null when momentum is 0).</summary>
    public abstract void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum);

    /// <summary>Adam: m, v moments updated in place; p -= lr * m / (sqrt(v) + eps). lr is already bias-corrected.</summary>
    public abstract void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps);

    public abstract void Synchronize();
}
