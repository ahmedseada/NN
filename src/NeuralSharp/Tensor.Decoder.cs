using NeuralSharp.Diagnostics;

namespace NeuralSharp;

// Operations of decoder-only language model layers.
public sealed partial class Tensor
{
    /// <summary>x / sqrt(mean(x²) + eps) over the last dimension (RMS normalization without a gain).</summary>
    internal Tensor RmsNormalize(float eps)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        var inv = Empty([rows], Device, track: false);
        Backend.RmsNorm(Storage, y.Storage, inv.Storage, rows, cols, eps);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rms_norm", g =>
            {
                x.Backend.RmsNormBackward(g.Storage, y.Storage, inv.Storage, x.GradStorage(), rows, cols);
                inv.Dispose();
            }, x);
        }
        else
        {
            inv.Dispose();
        }

        return Traced("rms_norm", y, start);
    }

    /// <summary>
    /// input · weights[j] (+ biases[j]) for every j in one pass over the input (inference: not recorded). The input is
    /// [..., k]; each weight is [k, n_j]; results are [..., n_j].
    /// </summary>
    internal static Tensor[] MatMulMany(Tensor input, IReadOnlyList<Tensor> weights, IReadOnlyList<Tensor?> biases)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[weights.Count];
        var products = new (Backends.Storage, Backends.Storage?, Backends.Storage, int)[weights.Count];
        for (int j = 0; j < weights.Count; j++)
        {
            int n = weights[j]._shape[1];
            outputs[j] = Empty([.. input._shape[..^1], n], input.Device);
            products[j] = (weights[j].Storage, biases[j]?.Storage, outputs[j].Storage, n);
        }

        input.Backend.MatMulMany(input.Storage, m, k, products);
        foreach (var output in outputs)
        {
            Traced("matmul_many", output, start);
        }

        return outputs;
    }

    /// <summary>
    /// <paramref name="forward"/>(input) without storing its intermediate results (activation checkpointing): the
    /// forward pass runs without recording, and the backward pass runs it again with recording, back-propagates through
    /// it and frees it at once. Parameters used inside receive their gradients then. Memory drops to the checkpointed
    /// outputs at the cost of a second forward pass. The function must be deterministic (no dropout).
    /// </summary>
    internal static Tensor Checkpoint(Func<Tensor, Tensor> forward, Tensor input)
    {
        if (!Autograd.IsEnabled)
        {
            return forward(input);
        }

        // Only the output survives the first pass: the module's intermediate results are freed at once.
        Tensor output;
        var seeds = new List<uint>();                                   // dropout masks are drawn again identically
        using (Layers.DropoutSeeds.Record(seeds))
        using (Autograd.NoGrad())
        using (var pass = new TensorScope())
        {
            output = pass.Keep(forward(input));
        }

        if (ReferenceEquals(output, input))
        {
            return output;
        }

        output.Record("checkpoint", g =>
        {
            using var scope = new TensorScope();
            var replay = input.Detach();
            replay.RequiresGrad = input.RequiresGrad;
            Tensor recomputed;
            using (Layers.DropoutSeeds.Replay(seeds))
            {
                recomputed = forward(replay);
            }

            if (recomputed.RequiresGrad)
            {
                recomputed.Backward(g);
            }

            if (input.RequiresGrad && replay.Grad is { } grad)
            {
                input.AddGradient(grad, adopt: true);             // the replay dies with this scope
            }
        }, input);
        return output;
    }

    /// <summary>
    /// Weighted token cross-entropy of a language model's output head applied to <paramref name="hidden"/> [rows, dim]:
    /// Σ_r w_r · (logsumexp(head(h_r)) - head(h_r)[t_r]) / <paramref name="normalizer"/>. The head runs on
    /// <paramref name="chunkRows"/> rows at a time and each chunk's gradient is back-propagated through the head at once,
    /// so the [rows, vocabulary] logits never exist together (a 2048-token sequence over a 152k vocabulary would need
    /// 1.2 GB for them and as much for their gradient). The result's backward pass adds the collected gradient to
    /// <paramref name="hidden"/>; parameters inside the head (adapters) receive theirs during this call, so back-propagate
    /// the loss unscaled (scale through <paramref name="normalizer"/> instead).
    /// </summary>
    internal static Tensor TokenCrossEntropy(Tensor hidden, Func<Tensor, Tensor> head, Tensor targets, Tensor weights, float normalizer, int chunkRows)
    {
        hidden.ThrowIfDisposed();
        if (hidden.Rank != 2 || targets.Size != hidden._shape[0] || weights.Size != hidden._shape[0])
        {
            throw new ArgumentException($"TokenCrossEntropy needs hidden [rows, dim] with rows targets and weights, got {FormatShape(hidden._shape)}, {targets.Size} and {weights.Size}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int rows = hidden._shape[0], dim = hidden._shape[1];
        chunkRows = Math.Max(1, chunkRows);
        var device = hidden.Device;
        var backend = hidden.Backend;
        bool record = WillRecord(hidden);
        var gradient = record ? Empty([rows, dim], device, zeroed: true) : null;
        var losses = Empty([rows], device);
        for (int r0 = 0; r0 < rows; r0 += chunkRows)
        {
            int n = Math.Min(chunkRows, rows - r0);
            using var scope = new TensorScope();
            var chunk = Empty([n, dim], device);
            backend.Copy2D(hidden.Storage, r0 * dim, n * dim, chunk.Storage, 0, n * dim, 1, n * dim, accumulate: false);
            chunk.RequiresGrad = record;
            var logits = head(chunk);
            int vocabulary = logits._shape[^1];
            if (logits.Size != n * vocabulary)
            {
                throw new ArgumentException($"The head must map [{n}, {dim}] to [{n}, vocabulary], got {FormatShape(logits._shape)}.");
            }

            var chunkTargets = Empty([n], device);
            var chunkWeights = Empty([n], device);
            var chunkLosses = Empty([n], device);
            backend.Copy2D(targets.Storage, r0, n, chunkTargets.Storage, 0, n, 1, n, accumulate: false);
            backend.Copy2D(weights.Storage, r0, n, chunkWeights.Storage, 0, n, 1, n, accumulate: false);
            backend.SoftmaxCrossEntropyRows(logits.Storage, chunkTargets.Storage, chunkWeights.Storage, chunkLosses.Storage, n, vocabulary,
                1f / normalizer);
            backend.Copy2D(chunkLosses.Storage, 0, n, losses.Storage, r0, n, 1, n, accumulate: false);
            if (record && logits.RequiresGrad)
            {
                logits.Backward(logits);                                         // the logits now hold their gradient
                backend.Copy2D(chunk.GradStorage(), 0, n * dim, gradient!.Storage, r0 * dim, n * dim, 1, n * dim, accumulate: false);
            }
        }

        var loss = Empty([1], device);
        backend.Sum(losses.Storage, loss.Storage, rows, 1f / normalizer);
        if (record)
        {
            loss.Record("token_cross_entropy", g => backend.Axpy(gradient!.Storage, hidden.GradStorage(), rows * dim, g.Item()), hidden);
        }

        return Traced("token_cross_entropy", loss, start);
    }

    /// <summary>
    /// (act(gate) · up) · W for a packed layer W with few rows, the activation applied as the input is read (not recorded;
    /// no bias), or null when the layer is not packed or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGated(Tensor gate, Tensor up, int activation, Layers.Linear layer)
    {
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        int kind = layer.Int8 is not null ? 0 : layer.Int4 is not null ? 1 : layer.BFloat16 is not null ? 2 : -1;
        int k = gate._shape[^1], m = gate.Size / Math.Max(1, k);
        if (kind < 0 || k != layer.InFeatures || up.Size != gate.Size || m > Backends.Cuda.PtxKernels.GemvRows)
        {
            return null;
        }

        var (packed, scales) = kind switch
        {
            0 => (layer.Int8!.Packed, layer.Int8.Scales),
            1 => (layer.Int4!.Packed, layer.Int4.Scales),
            _ => (layer.BFloat16!.Packed, (Tensor?)null),
        };
        if (packed.Device != gate.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. gate._shape[..^1], layer.OutFeatures], gate.Device);
        if (!gate.Backend.PackedMatMulGated(kind, activation, gate.Storage, up.Storage, packed.Storage, scales?.Storage, y.Storage, m, layer.OutFeatures, k))
        {
            y.Dispose();
            return null;
        }

        return Traced("matmul_gated_packed", y, start);
    }

    /// <summary>
    /// residual + x · W for a packed layer W with few rows (no bias, no adapter) and the RMS normalization of that sum
    /// with gain, in one device pass (inference: not recorded): the output projection of a block, its residual addition
    /// and the next normalization. Null when the layer or the shapes do not fit or the device has no fused version.
    /// </summary>
    internal static (Tensor Sum, Tensor Normalized)? MatMulPackedAddRmsNorm(Tensor x, Layers.Linear layer, Tensor residual, Layers.RMSNorm norm)
    {
        x.ThrowIfDisposed();
        residual.ThrowIfDisposed();
        int kind = layer.Int8 is not null ? 0 : layer.Int4 is not null ? 1 : layer.BFloat16 is not null ? 2 : -1;
        int k = x._shape[^1], m = x.Size / Math.Max(1, k), n = layer.OutFeatures;
        if (kind < 0 || layer.Bias is not null || layer.Adapter is not null || k != layer.InFeatures || m > Backends.Cuda.PtxKernels.GemvRows
            || residual.Size != m * n || residual._shape[^1] != n || norm.Features != n || x.Device.Type != DeviceType.Cuda)
        {
            return null;
        }

        var (packed, scales) = kind switch
        {
            0 => (layer.Int8!.Packed, layer.Int8.Scales),
            1 => (layer.Int4!.Packed, layer.Int4.Scales),
            _ => (layer.BFloat16!.Packed, (Tensor?)null),
        };
        if (packed.Device != x.Device || residual.Device != x.Device || norm.Gain.Device != x.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(residual._shape, x.Device, track: false);
        var sum = Empty(residual._shape, x.Device);
        var normalized = Empty(residual._shape, x.Device);
        bool done = x.Backend.PackedMatMulAddRmsNorm(kind, x.Storage, packed.Storage, scales?.Storage, y.Storage, m, n, k, residual.Storage, sum.Storage,
            norm.Gain.Storage, normalized.Storage, norm.Epsilon, norm.Offset);
        y.Dispose();                                                 // stream-ordered: freed after the kernel read it
        if (!done)
        {
            sum.Dispose();
            normalized.Dispose();
            return null;
        }

        return (sum, Traced("matmul_add_rms_norm_packed", normalized, start));
    }

    /// <summary>
    /// act(input · gate) · (input · up) for packed gate and up layers of one kind with few rows (no biases or adapters),
    /// the activation applied by the product's own last blocks (inference: not recorded), or null when the layers do
    /// not fit or the device has no fused version.
    /// </summary>
    internal static Tensor? MatMulPackedGatedPair(Tensor input, Layers.Linear gate, Layers.Linear up, int activation)
    {
        input.ThrowIfDisposed();
        int kind = gate.Int8 is not null ? 0 : gate.Int4 is not null ? 1 : gate.BFloat16 is not null ? 2 : -1;
        int k = input._shape[^1], m = input.Size / Math.Max(1, k), n = gate.OutFeatures;
        if (kind < 0 || m > Backends.Cuda.PtxKernels.GemvRows || input.Device.Type != DeviceType.Cuda || up.OutFeatures != n
            || gate.InFeatures != k || up.InFeatures != k || gate.Bias is not null || up.Bias is not null || gate.Adapter is not null
            || up.Adapter is not null || (kind == 0 ? up.Int8 is null : kind == 1 ? up.Int4 is null : up.BFloat16 is null))
        {
            return null;
        }

        (Tensor Packed, Tensor? Scales) Weights(Layers.Linear layer) => kind switch
        {
            0 => (layer.Int8!.Packed, layer.Int8.Scales),
            1 => (layer.Int4!.Packed, layer.Int4.Scales),
            _ => (layer.BFloat16!.Packed, null),
        };

        var (gatePacked, gateScales) = Weights(gate);
        var (upPacked, upScales) = Weights(up);
        if (gatePacked.Device != input.Device || upPacked.Device != input.Device)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var gateOut = Empty([m * n], input.Device, track: false);
        var upOut = Empty([m * n], input.Device, track: false);
        var hidden = Empty([.. input._shape[..^1], n], input.Device);
        bool done = input.Backend.PackedMatMulGatedPair(kind, activation, input.Storage, m, k,
            [(gatePacked.Storage, gateScales?.Storage, null, gateOut.Storage, n), (upPacked.Storage, upScales?.Storage, null, upOut.Storage, n)],
            hidden.Storage);
        gateOut.Dispose();                                           // stream-ordered: freed after the kernel read them
        upOut.Dispose();
        if (!done)
        {
            hidden.Dispose();
            return null;
        }

        return Traced("matmul_gated_pair_packed", hidden, start);
    }

    /// <summary>
    /// The layers' packed products of one input in one device pass (few rows, not recorded), or null when the device has
    /// no single-pass version.
    /// </summary>
    internal static Tensor[]? MatMulPackedMany(Tensor input, int kind, IReadOnlyList<Layers.Linear> layers)
    {
        input.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int k = input._shape[^1], m = input.Size / k;
        var outputs = new Tensor[layers.Count];
        var products = new (Backends.Storage, Backends.Storage?, Backends.Storage?, Backends.Storage, int)[layers.Count];
        for (int j = 0; j < layers.Count; j++)
        {
            var layer = layers[j];
            var (packed, scales) = kind switch
            {
                0 => (layer.Int8!.Packed, layer.Int8.Scales),
                1 => (layer.Int4!.Packed, layer.Int4.Scales),
                _ => (layer.BFloat16!.Packed, (Tensor?)null),
            };
            if (packed.Device != input.Device)
            {
                return null;
            }

            outputs[j] = Empty([.. input._shape[..^1], layer.OutFeatures], input.Device);
            products[j] = (packed.Storage, scales?.Storage, layer.Bias?.Storage, outputs[j].Storage, layer.OutFeatures);
        }

        if (!input.Backend.PackedMatMulMany(kind, input.Storage, m, k, products))
        {
            foreach (var output in outputs)
            {
                output.Dispose();
            }

            return null;
        }

        foreach (var output in outputs)
        {
            Traced("matmul_many_packed", output, start);
        }

        return outputs;
    }

    /// <summary>x / sqrt(mean(x²) + eps) · (gain + offset) over the last dimension in one pass (inference: not recorded).</summary>
    internal Tensor RmsNormAffine(Tensor gain, float eps, float offset)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = _shape[^1], rows = Size / cols;
        var y = Empty(_shape, Device);
        Backend.RmsNormAffine(Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return Traced("rms_norm_affine", y, start);
    }

    /// <summary>a + b (a residual addition) and its RMS normalization with gain, in one pass (inference: not recorded).</summary>
    internal static (Tensor Sum, Tensor Normalized) AddRmsNormAffine(Tensor a, Tensor b, Tensor gain, float eps, float offset)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(a, b);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int cols = a._shape[^1], rows = a.Size / cols;
        var sum = Empty(a._shape, a.Device);
        var y = Empty(a._shape, a.Device);
        a.Backend.AddRmsNormAffine(a.Storage, b.Storage, sum.Storage, gain.Storage, y.Storage, rows, cols, eps, offset);
        return (sum, Traced("add_rms_norm", y, start));
    }

    /// <summary>
    /// RMS normalization with gain of each head's vector of this [batch, steps, heads, dim] tensor, then the rotary
    /// embedding (see <see cref="Rope"/>), in one pass (inference: not recorded).
    /// </summary>
    internal Tensor RmsNormRope(Tensor gain, float eps, float offset, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        Backend.RmsNormRope(Storage, gain.Storage, cos.Storage, sin.Storage, positions.Storage, y.Storage, rows, dim, eps, offset, heads, steps,
            half, interleaved);
        return Traced("rms_norm_rope", y, start);
    }

    /// <summary><see cref="RmsNormRope"/> of the queries and the keys (same positions and tables) in one pass where the device can.</summary>
    internal static (Tensor Q, Tensor K) RmsNormRopePair(Tensor q, Tensor gainQ, float epsQ, float offsetQ, Tensor k, Tensor gainK, float epsK,
        float offsetK, Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = q._shape[1], dim = q._shape[3];
        var yq = Empty(q._shape, q.Device);
        var yk = Empty(k._shape, k.Device);
        q.Backend.RmsNormRopePair(q.Storage, gainQ.Storage, yq.Storage, q.Size / dim, epsQ, offsetQ, q._shape[2],
            k.Storage, gainK.Storage, yk.Storage, k.Size / dim, epsK, offsetK, k._shape[2], cos.Storage, sin.Storage, positions.Storage,
            dim, steps, half, interleaved);
        Traced("rms_norm_rope", yq, start);
        return (yq, Traced("rms_norm_rope", yk, start));
    }

    /// <summary>
    /// An attention layer's projections q [batch, steps, heads·dim], k and v [batch, steps, kvHeads·dim] in the layouts
    /// attention reads, in one pass where the device can (inference: not recorded): each query and key head
    /// RMS-normalized with gain (when the norms are given) and rotated (when the tables are given), queries as
    /// [batch·kvHeads, group·steps, dim] (the heads sharing a key/value head stacked), keys and values as
    /// [batch·kvHeads, steps, dim], or, with a cache, written into it at <paramref name="position"/> (then K and V are
    /// null). Null when the device has no fused version.
    /// </summary>
    internal static (Tensor Q, Tensor? K, Tensor? V)? NormRopeHeads(Tensor q, Tensor k, Tensor v, int heads, int kvHeads, int dim,
        Layers.RMSNorm? queryNorm, Layers.RMSNorm? keyNorm, Tensor? cos, Tensor? sin, Tensor positions, bool interleaved,
        Layers.KeyValueCache? cache, Tensor? position)
    {
        q.ThrowIfDisposed();
        k.ThrowIfDisposed();
        v.ThrowIfDisposed();
        if ((queryNorm is null) != (keyNorm is null) || (cos is null) != (sin is null) || (cache is null) != (position is null)
            || cache is { Format: Layers.KeyValueFormat.Int8 } || q.Device.Type != DeviceType.Cuda)
        {
            return null;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int n = q._shape[0], t = q._shape[1];
        var yq = Empty([n * kvHeads, heads / kvHeads * t, dim], q.Device);
        Tensor? yk = null, yv = null;
        int capacity = t, stride = dim;
        bool bfloat16 = cache is { Format: Layers.KeyValueFormat.BFloat16 };
        if (cache is null)
        {
            yk = Empty([n * kvHeads, t, dim], q.Device);
            yv = Empty([n * kvHeads, t, dim], q.Device);
        }
        else
        {
            capacity = cache.Keys._shape[1];
            stride = bfloat16 ? 2 * cache.Keys._shape[2] : cache.Keys._shape[2];
        }

        if (!q.Backend.NormRopeHeads(q.Storage, k.Storage, v.Storage, n, t, heads, kvHeads, dim, queryNorm?.Gain.Storage, queryNorm?.Epsilon ?? 0f,
            queryNorm?.Offset ?? 0f, keyNorm?.Gain.Storage, keyNorm?.Epsilon ?? 0f, keyNorm?.Offset ?? 0f, cos?.Storage, sin?.Storage, positions.Storage,
            cos?._shape[1] ?? 0, interleaved, yq.Storage, (yk ?? cache!.Keys).Storage, (yv ?? cache!.Values).Storage, position?.Storage, capacity, stride,
            bfloat16))
        {
            yq.Dispose();
            yk?.Dispose();
            yv?.Dispose();
            return null;
        }

        Traced("norm_rope_heads", yq, start);
        return (yq, yk, yv);
    }

    /// <summary>act(gate) · up element-wise (kind 0 = SiLU, 1 = GELU, 2 = ReLU), with its gradient: one pass either way.</summary>
    internal static Tensor GatedActivation(Tensor gate, Tensor up, int kind)
    {
        gate.ThrowIfDisposed();
        up.ThrowIfDisposed();
        CheckSameDevice(gate, up);
        if (!gate._shape.AsSpan().SequenceEqual(up._shape))
        {
            throw new ArgumentException($"Gate {FormatShape(gate._shape)} and up {FormatShape(up._shape)} differ.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(gate._shape, gate.Device);
        gate.Backend.GatedActivation(gate.Storage, up.Storage, y.Storage, gate.Size, kind);
        if (WillRecord(gate, up))
        {
            y.Record("gated_activation", g =>
            {
                int flags = (gate.RequiresGrad ? 1 : 0) | (up.RequiresGrad ? 2 : 0);
                gate.Backend.GatedActivationBackward(gate.Storage, up.Storage, g.Storage,
                    gate.RequiresGrad ? gate.GradStorage() : g.Storage, up.RequiresGrad ? up.GradStorage() : g.Storage, gate.Size, kind, flags);
            }, gate, up);
        }

        return Traced("gated_activation", y, start);
    }

    /// <summary>
    /// Rotary position embedding of this [batch, steps, heads, dim] tensor: pair p of each head's vector at step t
    /// rotates by the angle with cos/sin[positions[t], p] ([positions, half] tables). Dimensions beyond 2·half pass through.
    /// </summary>
    internal Tensor Rope(Tensor cos, Tensor sin, Tensor positions, int half, bool interleaved)
    {
        ThrowIfDisposed();
        if (Rank != 4)
        {
            throw new ArgumentException($"Rotary embedding expects [batch, steps, heads, dim], got {FormatShape(_shape)}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        int steps = _shape[1], heads = _shape[2], dim = _shape[3], rows = Size / dim;
        var y = Empty(_shape, Device);
        if (2 * half < dim)
        {
            Backend.Copy(Storage, y.Storage, Size);                        // dimensions beyond the rotated pairs pass through
        }

        Backend.Rope(Storage, y.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, 1f);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("rope", g =>
            {
                using var back = Empty(x._shape, x.Device, track: false);
                if (2 * half < dim)
                {
                    x.Backend.Copy(g.Storage, back.Storage, x.Size);
                }

                x.Backend.Rope(g.Storage, back.Storage, cos.Storage, sin.Storage, positions.Storage, rows, heads, steps, dim, half, interleaved, -1f);
                x.AddGradient(back, adopt: true);
            }, x);
        }

        return Traced("rope", y, start);
    }
}
