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

        PackedFewRows("int8_gemv_f32", x, q, scales, y, m, n, k, words);
    }

    public override void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        int words = (n + 1) / 2;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                BFloat16Dequantize(packed, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows("bf16_gemv_f32", x, packed, packed, y, m, n, k, words);
    }

    public override void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 7) / 8;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                Int4Dequantize(q, scales, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows("int4_gemv_f32", x, q, scales, y, m, n, k, words, align: 64);
    }

    public override void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 7) / 8;
        Launch1D(K("int4_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void BFloat16Dequantize(Storage packed, Storage w, int k, int n)
    {
        int words = (n + 1) / 2;
        Launch1D(K("bf16_dequant_f32"), k * words, P(packed), P(w), U(words), U(n), U(k * words));
    }

    // Few rows (decoding) through packed weights: read each weight word once, with enough blocks to keep every
    // multiprocessor busy; narrow matrices split k, and the last block of each column range adds the splits in order.
    // `align`: split boundaries fall on multiples of it (int4 splits start on a 64-row block).
    private void PackedFewRows(string kernel, Storage x, Storage q, Storage scales, Storage y, int m, int n, int k, int words, int align = 1)
    {
        int columnBlocks = (words + 31) / 32;
        int splits = Math.Clamp((4 * Math.Max(1, _multiprocessors) + columnBlocks - 1) / columnBlocks, 1, Math.Max(1, Math.Min(64, k / 64)));
        int chunk = ((k + splits - 1) / splits + align - 1) / align * align;
        splits = (k + chunk - 1) / chunk;
        var counters = SplitCounters(columnBlocks);
        if (splits == 1)
        {
            Launch(K(kernel), (uint)columnBlocks, 1, 1, PtxKernels.Int8GemvThreads, 1,
                P(x), P(q), P(scales), P(y), P(y), U(m), U(n), U(k), U(words), U(chunk), U(1), P(counters));
            return;
        }

        var part = Allocate(splits * m * n, zeroed: false);
        try
        {
            Launch(K(kernel), (uint)columnBlocks, (uint)splits, 1, PtxKernels.Int8GemvThreads, 1,
                P(x), P(q), P(scales), P(y), P(part), U(m), U(n), U(k), U(words), U(chunk), U(splits), P(counters));
        }
        finally
        {
            part.Release();
        }
    }

    private Storage? _splitCounters;

    // Zeroed arrival counters for split products, one per column block; the last block of a range resets its counter.
    private Storage SplitCounters(int blocks)
    {
        // A larger buffer replaces a smaller one without freeing it: recorded graphs may still use the old one.
        var current = Volatile.Read(ref _splitCounters);
        if (current is null || current.Length < blocks)
        {
            current = Allocate(Math.Max(blocks, 4096), zeroed: true);
            Volatile.Write(ref _splitCounters, current);
        }

        return current;
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
        DecodeSplit(rows, capacity, dim, y, (splits, part) => Launch(K("attention_decode_f32"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(position), P(y), P(part), U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(rows)));
    }

    // Decoding attention has one block per query row (few rows: the heads of one token), so the cached positions are
    // split over enough blocks to fill the GPU; each writes (max, sum, weighted values) for its chunk and
    // attention_combine_f32 merges them. The split count depends only on the shapes, so recorded graphs stay valid as
    // the cache fills (chunks are computed on the device from the current length).
    private void DecodeSplit(int rows, int capacity, int dim, Storage y, Action<int, Storage> launch)
    {
        int splits = Math.Clamp((2 * Math.Max(1, _multiprocessors) + rows - 1) / rows, 1, Math.Clamp(capacity / 64, 1, 32));
        if (splits == 1)
        {
            launch(1, y);
            return;
        }

        var part = Allocate(rows * splits * (dim + 2), zeroed: false);
        try
        {
            launch(splits, part);
            LaunchRows(K("attention_combine_f32"), rows, P(part), P(y), U(splits), U(dim), U(rows));
        }
        finally
        {
            part.Release();
        }
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));

    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionTiled(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale);
            return;
        }

        Launch(K("attention_flash_f32"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
            P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale));
    }

    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            ReadOnlySpan<ulong> args = [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale)];
            Launch(K("attn_bwd_kv_f32"), (uint)((capacity + 15) / 16), (uint)heads, 1, 128, 1, args);
            Launch(K("attn_bwd_q_f32"), (uint)((rowsPerHead + 31) / 32), (uint)heads, 1, 128, 1, args);
        }
        finally
        {
            delta.Release();
        }
    }

    public override void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled)
    {
        int words = (dim + 3) / 4;
        if (tiled && dim <= PtxKernels.FlashMaxDim)
        {
            Launch(K("attention_flash_int8"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale),
                P(keyScales), P(valueScales), U(words));
            return;
        }

        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Int8 cache attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(rows, capacity, dim, y, (splits, part) => Launch(K("attention_decode_int8"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(keyScales), P(valueScales), P(position), P(y), P(part),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words), U(rows)));
    }

    public override void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("add_rms_norm_affine_f32"), rows, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope_f32"), rows, P(x), P(gain), P(cos), P(sin), P(positions), P(y),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows));
}
