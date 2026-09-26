namespace NeuralSharp.Backends.Cuda;

// Int8 weight-only quantization kernels (see PtxKernels.Quantized.cs).
internal sealed unsafe partial class CudaBackend
{
    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 3) / 4;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            Launch1D(K("int8_matmul_f32"), m * words, P(x), P(q), P(scales), P(y), U(k), U(words), U(n), U(m * words));
            return;
        }

        // Few rows (decoding): read each weight word once, with enough blocks to keep every multiprocessor busy.
        // Narrow matrices split k into chunks whose partial sums are added in order by a second kernel.
        int columnBlocks = (words + 31) / 32;
        int splits = Math.Clamp((4 * Math.Max(1, _multiprocessors) + columnBlocks - 1) / columnBlocks, 1, Math.Max(1, Math.Min(64, k / 64)));
        int chunk = (k + splits - 1) / splits;
        splits = (k + chunk - 1) / chunk;
        if (splits == 1)
        {
            Launch(K("int8_gemv_f32"), (uint)columnBlocks, 1, 1, PtxKernels.Int8GemvThreads, 1,
                P(x), P(q), P(scales), P(y), P(y), U(m), U(n), U(k), U(words), U(chunk), U(1));
            return;
        }

        var part = Allocate(splits * m * n, zeroed: false);
        try
        {
            Launch(K("int8_gemv_f32"), (uint)columnBlocks, (uint)splits, 1, PtxKernels.Int8GemvThreads, 1,
                P(x), P(q), P(scales), P(y), P(part), U(m), U(n), U(k), U(words), U(chunk), U(splits));
            Launch1D(K("int8_gemv_finish_f32"), m * n, P(part), P(scales), P(y), U(n), U(m * n), U(splits), U(m * n));
        }
        finally
        {
            part.Release();
        }
    }

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 3) / 4;
        Launch1D(K("int8_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        int n = heads * steps;
        Launch1D(K("kv_write_int8"), n, P(source), P(cache), P(scales), P(position), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int n = rows * steps * capacity;
        Launch1D(K("attn_scores_int8"), n, P(q), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4, n = rows * steps * words;
        Launch1D(K("attn_context_int8"), n, P(weights), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U(words), U(n));
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps) =>
        LaunchRows(K("rms_norm_f32"), rows, P(x), P(y), P(inv), U(cols), F(eps), U(rows));

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols) =>
        LaunchRows(K("rms_norm_backward_f32"), rows, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        int n = rows * half;
        Launch1D(K("rope_f32"), n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign), U(n));
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Decoding attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        LaunchRows(K("attention_decode_f32"), rows, P(q), P(keys), P(values), P(position), P(y),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(rows));
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));
}
