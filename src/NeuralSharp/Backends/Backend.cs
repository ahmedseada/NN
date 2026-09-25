namespace NeuralSharp.Backends;

internal enum UnaryOp
{
    Sigmoid,
    Tanh,
    Relu,
    Square,
    Abs,
    Exp,
    Log,
    Gelu,
}

internal enum BinaryOp
{
    Add,
    Sub,
    Mul,
}

/// <summary>Shape parameters of a 2-D convolution or pooling window over NCHW data.</summary>
internal readonly record struct ConvGeometry(
    int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW)
{
    public int OH => (H + 2 * PH - KH) / SH + 1;

    public int OW => (W + 2 * PW - KW) / SW + 1;

    /// <summary>Columns of the unfolded matrix: C * KH * KW.</summary>
    public int PatchSize => C * KH * KW;

    /// <summary>Rows of the unfolded matrix: N * OH * OW.</summary>
    public int Positions => N * OH * OW;
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

    public abstract MemoryUsage GetMemoryUsage();

    /// <summary>Frees pooled blocks that no tensor is using.</summary>
    public abstract void ReleaseCachedMemory();

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

    /// <summary>y[offset] += alpha * x[0] (used to accumulate scalar losses without leaving the device).</summary>
    public abstract void AxpyAt(Storage x, Storage y, int offset, float alpha);

    /// <summary>y[i] += scale * s[0].</summary>
    public abstract void AddBroadcastScalar(Storage s, Storage y, int n, float scale);

    /// <summary>
    /// c[m, n] = op(a) * op(b) + beta * c, where op(a) is [m, k] and op(b) is [k, n].
    /// With <paramref name="transA"/> a is stored as [k, m]; with <paramref name="transB"/> b is stored as [n, k].
    /// </summary>
    public void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta) =>
        BatchedMatMul(a, b, c, 1, m, n, k, transA, transB, beta);

    /// <summary><see cref="MatMul"/> for <paramref name="batch"/> independent, contiguous matrix triples.</summary>
    public abstract void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta);

    /// <summary>Row-wise softmax (or log-softmax) over the last dimension: y[r, :] = softmax(x[r, :]).</summary>
    public abstract void Softmax(Storage x, Storage y, int rows, int cols, bool log);

    /// <summary>
    /// Softmax: dx += y * (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) * Σ dy. Sums are per row; y is the forward output.
    /// </summary>
    public abstract void SoftmaxBackward(Storage y, Storage dy, Storage dx, int rows, int cols, bool log);

    /// <summary>y[r] = index of the largest element of row r.</summary>
    public abstract void ArgMax(Storage x, Storage y, int rows, int cols);

    /// <summary>
    /// y[r] = 1 when the prediction in row r is right, else 0. For one column: (p ≥ threshold) == (t ≥ 0.5);
    /// otherwise argmax(p) == argmax(t).
    /// </summary>
    public abstract void ClassMatch(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold);

    // Grouped normalization. Data is viewed as [outer, groups, inner]; group g holds the M = outer * inner
    // elements x[o, g, i]. BatchNorm uses groups = channels; LayerNorm uses outer = 1, groups = rows, inner = features.

    /// <summary>Per-group mean, (biased) variance and 1 / sqrt(variance + eps).</summary>
    public abstract void NormStats(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps);

    /// <summary>y = (x - mean[g]) * invStd[g].</summary>
    public abstract void NormApply(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner);

    /// <summary>dx += invStd[g] / M * (M * dxhat - sum1[g] - xhat * sum2[g]).</summary>
    public abstract void NormBackward(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner);

    /// <summary>
    /// y (+)= x * scale[g] + shift[g] with g = (i / inner) % groups; a null scale means 1, a null shift means 0.
    /// </summary>
    public abstract void GroupScaleShift(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate);

    /// <summary>sumA[g] += Σ a; sumAB[g] += Σ a·b over each group (see NormStats for the layout). b/sumAB may be null.</summary>
    public abstract void GroupReduce(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner);

    /// <summary>y = 1 / sqrt(x + eps).</summary>
    public abstract void InvSqrt(Storage x, Storage y, int n, float eps);

    /// <summary>Embedding lookup: y[i, :] = table[indices[i], :] for count indices of width dim.</summary>
    public abstract void Gather(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary);

    /// <summary>dtable[indices[i], :] += dy[i, :].</summary>
    public abstract void ScatterAdd(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary);

    /// <summary>Unfolds image patches: cols[(n, oh, ow), (c, kh, kw)] = x[n, c, oh*sh - ph + kh, ow*sw - pw + kw] (0 outside).</summary>
    public abstract void Im2Col(Storage x, Storage cols, in ConvGeometry g);

    /// <summary>The adjoint of <see cref="Im2Col"/>: dx += fold(dcols).</summary>
    public abstract void Col2Im(Storage dcols, Storage dx, in ConvGeometry g);

    /// <summary>Max pooling; argmax receives the flat input index of each maximum (as raw int bits).</summary>
    public abstract void MaxPool(Storage x, Storage y, Storage argmax, in ConvGeometry g);

    /// <summary>dx[argmax[i]] += dy[i].</summary>
    public abstract void MaxPoolBackward(Storage dy, Storage argmax, Storage dx, int count);

    /// <summary>
    /// y (+)= x permuted: output element at coordinates (c0..c[r-1]) of <paramref name="outShape"/> comes from
    /// input offset Σ c_k * inStrides[k] (the input strides already reordered by the permutation). Rank ≤ 6.
    /// </summary>
    public abstract void Permute(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate);

    /// <summary>
    /// Strided block copy: dst[dstOffset + r*dstStride + c] (+)= src[srcOffset + r*srcStride + c] for r &lt; rows, c &lt; cols.
    /// Implements narrow, concatenation and their gradients.
    /// </summary>
    public abstract void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate);

    /// <summary>y[o, i] (+)= scale * Σ_d x[o, d, i] for a [outer, dim, inner] view.</summary>
    public abstract void SumAxis(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate);

    /// <summary>dx[o, d, i] += scale * dy[o, i] (the gradient of <see cref="SumAxis"/>).</summary>
    public abstract void BroadcastAxis(Storage dy, Storage dx, int outer, int dim, int inner, float scale);

    /// <summary>SGD with optional momentum: v = momentum * v + g; p -= lr * v (v is null when momentum is 0).</summary>
    public abstract void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum);

    /// <summary>Adam: m, v moments updated in place; p -= lr * m / (sqrt(v) + eps). lr is already bias-corrected.</summary>
    public abstract void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps);

    /// <summary>Inverted dropout: y = keep(i) ? x / (1 - p) : 0, where keep(i) comes from <see cref="DropoutMask"/>.</summary>
    public abstract void Dropout(Storage x, Storage y, int n, float p, uint seed);

    /// <summary>dx += keep(i) ? dy / (1 - p) : 0, regenerating the same mask from the seed.</summary>
    public abstract void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed);

    public abstract void Synchronize();
}

/// <summary>
/// Stateless per-element random mask for dropout, identical on every backend: a MurmurHash3
/// finalizer of (index * golden ratio) ^ seed gives 24 random bits. Because the mask is a pure
/// function of (seed, index) it never has to be stored for the backward pass.
/// </summary>
internal static class DropoutMask
{
    public static bool Keep(uint seed, uint index, float p)
    {
        uint h = (index * 0x9E3779B9u) ^ seed;
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f) >= p;
    }
}

/// <summary>Thread-safe byte accounting shared by the backends' caching allocators.</summary>
internal sealed class MemoryAccountant(Func<long?> limit, string deviceName)
{
    private long _inUse;
    private long _cached;

    public MemoryUsage Usage => new(Interlocked.Read(ref _inUse), Interlocked.Read(ref _cached), limit());

    /// <summary>
    /// Called before allocating <paramref name="bytes"/> of new memory. Returns true when the cache
    /// should be released first to stay under the limit; throws when even that is not enough.
    /// </summary>
    public bool MustReleaseCacheFor(long bytes)
    {
        if (limit() is not { } max)
        {
            return false;
        }

        long inUse = Interlocked.Read(ref _inUse);
        if (inUse + bytes > max)
        {
            throw new ResourceLimitExceededException(
                $"Allocating {bytes:N0} bytes on {deviceName} would exceed its memory limit of {max:N0} bytes ({inUse:N0} in use). " +
                "Raise the limit in ComputeResources, use smaller batches, or dispose tensors you no longer need.");
        }

        return inUse + Interlocked.Read(ref _cached) + bytes > max;
    }

    public void Allocated(long bytes) => Interlocked.Add(ref _inUse, bytes);

    public void Reused(long bytes)
    {
        Interlocked.Add(ref _cached, -bytes);
        Interlocked.Add(ref _inUse, bytes);
    }

    public void Returned(long bytes)
    {
        Interlocked.Add(ref _inUse, -bytes);
        Interlocked.Add(ref _cached, bytes);
    }

    public void Freed(long bytes) => Interlocked.Add(ref _cached, -bytes);
}
