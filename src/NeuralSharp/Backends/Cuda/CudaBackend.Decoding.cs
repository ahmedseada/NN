using static NeuralSharp.Backends.Cuda.CudaDriver;

namespace NeuralSharp.Backends.Cuda;

// Fused inference kernels, incremental-decoding kernels and CUDA Graph capture/replay.
internal sealed unsafe partial class CudaBackend
{
    public override void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale) =>
        Launch1D(K("scale_mask_softmax_f32"), rows, P(x), mask is null ? P(x) : P(mask), P(y),
            U(cols), U(Math.Max(maskRows, 1)), F(scale), U(mask is null ? 0 : 1), U(rows));

    public override void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps) =>
        Launch1D(K("layernorm_fused_f32"), rows, P(x), P(gamma), P(beta), P(y), U(cols), F(eps), U(rows));

    public override void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols) =>
        Launch1D(K("bias_gelu_f32"), n, P(x), P(bias), P(y), U(cols), U(n));

    public override void DecoderMask(Storage position, Storage mask, int rows, int capacity)
    {
        int n = rows * capacity;
        Launch1D(K("decoder_mask_f32"), n, P(position), P(mask), U(capacity), U(n));
    }

    public override void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int n = heads * steps * dim;
        Launch1D(K("kv_write_f32"), n, P(source), P(cache), P(position), U(steps * dim), U(capacity * dim), U(dim), U(n));
    }

    public override void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed) =>
        Launch1D(K("sample_rows_f32"), rows, P(logits), P(ids), P(stats), P(step),
            U(vocabulary), U(rowStride), U(rowOffset), F(1f / MathF.Max(temperature, 1e-3f)), U(topK), F(topP), F(minP), seed, U(rows), U(rows));

    public override void PenalizeRows(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency) =>
        Launch1D(K("penalize_rows_f32"), rows, P(logits), P(work), P(history), P(length),
            U(vocabulary), U(rowStride), U(rowOffset), U(capacity), U(Math.Min(lastN, capacity)), F(repeat), F(presence), F(frequency), U(rows), U(rows));

    public override void HistoryPush(Storage ids, Storage history, Storage length, int rows, int capacity) =>
        Launch1D(K("history_push_f32"), rows, P(ids), P(history), P(length), U(capacity), U(rows), U(rows));

    public override bool SupportsGraphs => true;

    public override void BeginCapture()
    {
        MakeCurrent();
        lock (_pool)
        {
            if (_captureFree is not null)
            {
                throw new InvalidOperationException("A graph is already being recorded on this device.");
            }

            _captureFree = [];
        }

        int result = cuStreamBeginCapture(_stream, StreamCaptureModeRelaxed);
        if (result != 0)
        {
            lock (_pool)
            {
                _captureFree = null;
            }

            Check(result, nameof(cuStreamBeginCapture));
        }
    }

    public override (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture()
    {
        MakeCurrent();
        var owned = TakeCaptureBlocks();
        Check(cuStreamEndCapture(_stream, out IntPtr graph), nameof(cuStreamEndCapture));
        int result = cuGraphInstantiateWithFlags(out IntPtr executable, graph, 0);
        if (result != 0)
        {
            cuGraphDestroy(graph);
            owned.ForEach(o => o.Release());
            Check(result, nameof(cuGraphInstantiateWithFlags));
        }

        return (executable, graph, owned);
    }

    public override List<Storage> AbortCapture()
    {
        MakeCurrent();
        var owned = TakeCaptureBlocks();
        if (cuStreamEndCapture(_stream, out IntPtr graph) == 0 && graph != IntPtr.Zero)
        {
            cuGraphDestroy(graph);
        }

        return owned;
    }

    /// <summary>Ends capture-mode allocation and wraps the blocks freed during capture as storages the graph owns.</summary>
    private List<Storage> TakeCaptureBlocks()
    {
        lock (_pool)
        {
            var owned = new List<Storage>();
            foreach (var (length, bucket) in _captureFree ?? [])
            {
                foreach (ulong pointer in bucket)
                {
                    // Re-count them as in use; releasing the storage later returns them to the shared pool.
                    _memory.Reused(BlockBytes(length));
                    owned.Add(new CudaStorage(this, pointer, length));
                }
            }

            _captureFree = null;
            return owned;
        }
    }

    public override void ReplayGraph(IntPtr executable)
    {
        MakeCurrent();
        Check(cuGraphLaunch(executable, _stream), nameof(cuGraphLaunch));
    }

    public override void DestroyGraph(IntPtr executable, IntPtr graph)
    {
        MakeCurrent();
        cuGraphExecDestroy(executable);
        cuGraphDestroy(graph);
    }
}
