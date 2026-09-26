namespace NeuralSharp.Backends.Cuda;

// Int8 weight-only quantization kernels (see PtxKernels.Quantized.cs).
internal sealed unsafe partial class CudaBackend
{
    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 3) / 4;
        Launch1D(K("int8_matmul_f32"), m * words, P(x), P(q), P(scales), P(y), U(k), U(words), U(n), U(m * words));
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
}
