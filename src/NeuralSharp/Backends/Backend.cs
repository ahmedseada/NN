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

    /// <summary>Copies destination.Length floats starting at element <paramref name="offset"/>.</summary>
    public abstract void DownloadRange(Storage source, int offset, Span<float> destination);

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

    /// <summary>
    /// y = x · w for many rows (prompts) with packed weights w (<paramref name="kind"/> as in
    /// <see cref="PackedMatMulMany"/>), expanding w as it is read instead of into a float copy. Returns false when the
    /// device has no such kernel or the shape is too small for it (callers then expand w first).
    /// </summary>
    public virtual bool PackedMatMulLarge(int kind, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k) => false;

    /// <summary>
    /// y = (act(gate) · up) · w for few rows with packed weights w (<paramref name="kind"/> as in
    /// <see cref="PackedMatMulMany"/>; activation 0 = SiLU, 1 = GELU tanh): the gated feed-forward's down projection
    /// without a separate activation pass. Returns false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulGated(int kind, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k) => false;

    /// <summary>
    /// Several few-row products of packed weights sharing one input x [m, k]: y_j = x · w_j (+ bias_j), with w_j int8
    /// (<paramref name="kind"/> 0, as in <see cref="Int8MatMul"/>), 4-bit (1, <see cref="Int4MatMul"/>) or bfloat16 (2,
    /// <see cref="BFloat16MatMul"/>; no scales). Returns false when the device has no single-pass version (callers then
    /// run the products one by one).
    /// </summary>
    public virtual bool PackedMatMulMany(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products) => false;

    /// <summary>
    /// Several products sharing one input: y_j = a · w_j (+ bias_j) for a [m, k] and w_j [k, n_j] (the query, key and
    /// value projections, say). The default computes them one by one; devices may do them in one pass.
    /// </summary>
    public virtual void MatMulMany(Storage a, int m, int k, ReadOnlySpan<(Storage Weight, Storage? Bias, Storage Output, int Columns)> products)
    {
        foreach (var (weight, bias, output, columns) in products)
        {
            MatMul(a, weight, output, m, columns, k, false, false, 0f);
            if (bias is not null)
            {
                AddRowVector(output, bias, output, m, columns);
            }
        }
    }

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

    /// <summary><see cref="Gather"/> from a bfloat16 table packed as in <see cref="BFloat16MatMul"/> ([vocabulary, dim]).</summary>
    public abstract void GatherBFloat16(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary);

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

    // ---------------------------------------------------------------- fused inference kernels

    /// <summary>y[r, :] = softmax(scale * x[r, :] + mask[r % maskRows, :]) (mask optional).</summary>
    public abstract void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale);

    /// <summary>y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dimension.</summary>
    public abstract void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps);

    /// <summary>y[i] = gelu(x[i] + bias[i % cols]).</summary>
    public abstract void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols);

    /// <summary>
    /// y[m, n] = x[m, k] · w, where w[k, j] = q[k, j] · scales[j] and q holds signed bytes packed four per 32-bit element
    /// along each row (rows padded to ceil(n / 4) elements). Suited to few rows (token-by-token decoding).
    /// </summary>
    public abstract void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k);

    /// <summary>
    /// y[m, n] = x[m, k] · w[k, n] with w stored as bfloat16 pairs packed into 32-bit words along each row (word c of
    /// row r holds columns 2c in its low half and 2c + 1 in its high half; rows have ⌈n / 2⌉ words).
    /// </summary>
    public abstract void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k);

    /// <summary>w[k, n] = the float32 values of bfloat16 weights packed as in <see cref="BFloat16MatMul"/>.</summary>
    public abstract void BFloat16Dequantize(Storage packed, Storage w, int k, int n);

    /// <summary>w[k, n] = q[k, n] · scales[n] (see <see cref="Int8MatMul"/> for the packing).</summary>
    public abstract void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n);

    /// <summary>
    /// y[m, n] = x[m, k] · w with 4-bit weights: w[r, j] = q[r, j] · scales[r / 32, j], where q holds signed nibbles packed
    /// eight per 32-bit word along each row (nibble c of word w is column 8w + c; rows have ⌈n / 8⌉ words) and scales has
    /// one row of 8·⌈n / 8⌉ values per group of 32 weight rows.
    /// </summary>
    public abstract void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k);

    /// <summary>w[k, n] = the float values of 4-bit weights packed as in <see cref="Int4MatMul"/>.</summary>
    public abstract void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n);

    // Int8 KV cache: each cached row (one head, one position) is dim bytes packed four per element, words = ceil(dim / 4)
    // elements, with one scale per row in scales[head, position].

    /// <summary>y = x · inv per row, inv[r] = 1 / sqrt(mean(x[r]²) + eps) (stored for the backward pass).</summary>
    public abstract void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps);

    /// <summary>dx += inv[r] · (dy - y · mean(dy · y)) per row, where y is the normalized forward output.</summary>
    public abstract void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols);

    /// <summary>
    /// Rotary position embedding of x [rows = batch·steps·heads, dim] into y (which must already hold x): pair p of a row at
    /// step t rotates by the angle whose cos/sin are cos/sin[positions[t], p]. Pairs are (2p, 2p+1) when interleaved,
    /// else (p, p + half). sign = -1 rotates backwards (the gradient).
    /// </summary>
    public abstract void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign);

    /// <summary>y = x · inv · (gain[c] + offset) per row with inv = 1 / sqrt(mean(x²) + eps): normalization and gain in one pass (inference).</summary>
    public abstract void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset);

    /// <summary>sum = a + b and y = RMS-normalized sum · (gain[c] + offset) per row, in one pass (inference).</summary>
    public abstract void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset);

    /// <summary>
    /// RMS normalization with gain of each row (a head's vector), then the rotary embedding as <see cref="Rope"/> with
    /// sign 1 (rows are batch·steps·heads; dimensions beyond 2·half are only normalized). Inference.
    /// </summary>
    public abstract void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved);

    /// <summary>
    /// <see cref="RmsNormRope"/> for two tensors sharing the positions and rotary tables (queries and keys): rows1 rows
    /// of x (heads per step: heads) and rows2 of x2 (heads2). Devices may do both in one pass.
    /// </summary>
    public virtual void RmsNormRopePair(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved)
    {
        RmsNormRope(x, gain, cos, sin, positions, y, rows1, cols, eps, offset, heads, steps, half, interleaved);
        RmsNormRope(x2, gain2, cos, sin, positions, y2, rows2, cols, eps2, offset2, heads2, steps, half, interleaved);
    }

    /// <summary>
    /// Token cross-entropy for language-model training, per row r of logits [rows, vocabulary] with target class t_r and
    /// weight w_r (0 masks the row): losses[r] = w_r · (logsumexp(x_r) - x_r[t_r]); the logits are overwritten with their
    /// gradient scale · w_r · (softmax(x_r) - onehot(t_r)). Targets and weights are float arrays of rows.
    /// </summary>
    public abstract void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale);

    /// <summary>y = act(gate) · up element-wise; kind 0 = SiLU, 1 = GELU (tanh approximation), 2 = ReLU.</summary>
    public abstract void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind);

    /// <summary>dgate += dy · up · act'(gate) when flags has bit 0, dup += dy · act(gate) when it has bit 1.</summary>
    public abstract void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags);

    /// <summary>Quantizes source [heads·steps, dim] into the int8 cache at positions position[0] + step.</summary>
    public abstract void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>y[r, t, c] = scales[r, c] · Σ_d q[r, t, d] · keys[r, c, d] for every cached position c.</summary>
    public abstract void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim);

    /// <summary>y[r, t, d] = Σ_c weights[r, t, c] · scales[r, c] · values[r, c, d].</summary>
    public abstract void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim);

    /// <summary>
    /// Attention over a key/value cache filled up to the device position: for head h and row i of q [heads, rowsPerHead,
    /// dim], y[h, i] = Σ_c softmax(scale · q[h, i] · keys[h, c]) · values[h, c] over positions c = 0 … position[0] +
    /// (i % steps) (the causal limit of that row's step). Keys and values are [heads, capacity, dim]; unfilled
    /// positions are never read, so the cost follows the context length rather than the capacity.
    /// </summary>
    public abstract void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale);

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>, for many query rows) over an int8 cache: keys and
    /// values [heads, capacity, ⌈dim / 4⌉ words] of packed bytes with one scale per cached row (keyScales, valueScales
    /// [heads, capacity]).
    /// </summary>
    public abstract void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled);

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>) over a bfloat16 cache: keys and values
    /// [heads, capacity, ⌈dim / 2⌉ words], dimension d in the low (even d) or high (odd d) half of word d / 2.
    /// </summary>
    public abstract void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled);

    /// <summary><see cref="KeyValueWrite"/> into a bfloat16 cache (values rounded to nearest, ties to even).</summary>
    public abstract void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>
    /// Gradient of <see cref="AttentionTiled"/> with causal offset 0 (training): given the output, each row's log-sum-exp
    /// and dOutput, adds to dq [heads, rowsPerHead, dim] and dkeys, dvalues [heads, capacity, dim]. The attention weights
    /// are recomputed, never stored.
    /// </summary>
    public abstract void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale);

    /// <summary>
    /// The same attention as <see cref="AttentionDecode"/> for many query rows at once (a prompt, a training sequence),
    /// tiled so query rows share each key and value read; also writes each row's log-sum-exp of the scaled scores to
    /// <paramref name="logSumExp"/> [heads, rowsPerHead] when given.
    /// </summary>
    public virtual void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale) =>
        AttentionDecode(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale);

    // ---------------------------------------------------------------- incremental decoding (positions live on the device)

    /// <summary>mask[i, j] = j ≤ position + i ? 0 : -1e9 for a [rows, capacity] mask; position is read from device memory.</summary>
    public abstract void DecoderMask(Storage position, Storage mask, int rows, int capacity);

    /// <summary>cache[bh, position + t, :] = source[bh, t, :] for [heads, steps, dim] → [heads, capacity, dim].</summary>
    public abstract void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim);

    /// <summary>
    /// Draws one token per row from softmax(logits / temperature), restricted in turn to the top-k scores (topK &gt; 0),
    /// to the smallest set holding topP of the probability (0 &lt; topP &lt; 1; the cut-off is found by bisection on the
    /// score) and to tokens at least minP times as likely as the best one (minP &gt; 0), using the counter-based random
    /// stream (seed, step, row). Writes the token to ids[row] and 13 statistics to stats[(step * rows + row) * 13]:
    /// id, probability, entropy (bits), then the top-5 (id, probability) pairs. The step number is read from device
    /// memory. Row r's logits start at element r * rowStride + rowOffset.
    /// </summary>
    public abstract void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed);

    /// <summary>
    /// Copies each row's logits (row r at r * rowStride + rowOffset) to work[r, :] and applies repetition penalties for
    /// the last min(length, lastN) tokens of the row's history ring history[r, pos % capacity] (length read from device
    /// memory). Each distinct token in the window is penalized once: repeat (x &gt; 0 ? x / repeat : x * repeat), then
    /// x -= presence + frequency * (occurrences in the window).
    /// </summary>
    public abstract void PenalizeRows(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency);

    /// <summary>history[r, length % capacity] = ids[r] for every row (length read from device memory, not advanced).</summary>
    public abstract void HistoryPush(Storage ids, Storage history, Storage length, int rows, int capacity);

    // ---------------------------------------------------------------- graphs

    /// <summary>Whether this device can record and replay work as a graph.</summary>
    public virtual bool SupportsGraphs => false;

    /// <summary>Starts recording all subsequent work on this device instead of running it.</summary>
    public virtual void BeginCapture() => throw new NotSupportedException();

    /// <summary>Stops recording and returns a replayable graph plus the device blocks it owns.</summary>
    public virtual (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture() => throw new NotSupportedException();

    /// <summary>Stops recording after a failure, discarding the partial graph.</summary>
    public virtual List<Storage> AbortCapture() => [];

    public virtual void ReplayGraph(IntPtr executable) => throw new NotSupportedException();

    public virtual void DestroyGraph(IntPtr executable, IntPtr graph)
    {
    }
}

/// <summary>Stateless counter-based random numbers shared by every backend (the sampler's random stream).</summary>
internal static class CounterRandom
{
    /// <summary>A uniform float in [0, 1) from (seed, step, row) via the MurmurHash3 finalizer.</summary>
    public static float Uniform(uint seed, uint step, uint row)
    {
        uint h = seed ^ (step * 0x9E3779B9u) ^ (row * 0x85EBCA6Bu);
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f);
    }
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
    private long _offloaded;

    public MemoryUsage Usage => new(Interlocked.Read(ref _inUse), Interlocked.Read(ref _cached), limit(), Interlocked.Read(ref _offloaded));

    /// <summary>Counts bytes placed in system memory for this device (negative when released).</summary>
    public void Offloaded(long bytes) => Interlocked.Add(ref _offloaded, bytes);

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
