namespace NeuralSharp.Backends.Cpu;

// Fused inference kernels and incremental-decoding primitives (KV cache, masks, on-device sampling).
internal sealed partial class CpuBackend
{
    public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
        D(source).AsSpan(offset, destination.Length).CopyTo(destination);

    public override void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        float[] xv = D(x), yv = D(y);
        float[]? mv = mask is null ? null : D(mask);
        For(rows, (long)rows * cols * 8, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                var ms = mv is null ? default : mv.AsSpan(r % maskRows * cols, cols);
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    ys[j] = xs[j] * scale + (mv is null ? 0f : ms[j]);
                    max = MathF.Max(max, ys[j]);
                }

                float sum = 0f;
                for (int j = 0; j < cols; j++)
                {
                    ys[j] = MathF.Exp(ys[j] - max);
                    sum += ys[j];
                }

                float inv = 1f / sum;
                for (int j = 0; j < cols; j++)
                {
                    ys[j] *= inv;
                }
            }
        });
    }

    public override void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        float[] xv = D(x), gv = D(gamma), bv = D(beta), yv = D(y);
        For(rows, (long)rows * cols * 4, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                var xs = xv.AsSpan(r * cols, cols);
                var ys = yv.AsSpan(r * cols, cols);
                float mean = 0f;
                foreach (float v in xs)
                {
                    mean += v;
                }

                mean /= cols;
                float var = 0f;
                foreach (float v in xs)
                {
                    var += (v - mean) * (v - mean);
                }

                float inv = 1f / MathF.Sqrt(var / cols + eps);
                for (int j = 0; j < cols; j++)
                {
                    ys[j] = (xs[j] - mean) * inv * gv[j] + bv[j];
                }
            }
        });
    }

    public override void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols)
    {
        float[] xv = D(x), bv = D(bias), yv = D(y);
        For(n / cols, n * 8L, (start, end) =>
        {
            for (int r = start; r < end; r++)
            {
                int o = r * cols;
                for (int j = 0; j < cols; j++)
                {
                    float v = xv[o + j] + bv[j];
                    yv[o + j] = 0.5f * v * (1f + MathF.Tanh(GeluK * (v + GeluC * v * v * v)));
                }
            }
        });
    }

    public override void DecoderMask(Storage position, Storage mask, int rows, int capacity)
    {
        int pos = (int)D(position)[0];
        float[] mv = D(mask);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < capacity; j++)
            {
                mv[i * capacity + j] = j <= pos + i ? 0f : -1e9f;
            }
        }
    }

    public override void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int pos = (int)D(position)[0];
        float[] sv = D(source), cv = D(cache);
        for (int h = 0; h < heads; h++)
        {
            sv.AsSpan(h * steps * dim, steps * dim).CopyTo(cv.AsSpan((h * capacity + pos) * dim, steps * dim));
        }
    }

    public override void SampleRows(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, uint seed)
    {
        float[] lv = D(logits), iv = D(ids), sv = D(stats);
        uint stepNumber = (uint)D(step)[0];
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        Span<float> e = vocabulary <= 4096 ? stackalloc float[vocabulary] : new float[vocabulary];
        Span<int> taken = stackalloc int[5];
        for (int r = 0; r < rows; r++)
        {
            var z = lv.AsSpan(r * rowStride + rowOffset, vocabulary);
            float max = float.NegativeInfinity;
            foreach (float v in z)
            {
                max = MathF.Max(max, v * invT);
            }

            // Top-k: the k-th largest distinct scaled score is the cut-off (ties at the cut-off are kept).
            float threshold = float.NegativeInfinity;
            if (topK > 0 && topK < vocabulary)
            {
                threshold = float.PositiveInfinity;
                for (int k = 0; k < topK; k++)
                {
                    float next = float.NegativeInfinity;
                    foreach (float v in z)
                    {
                        float s = v * invT;
                        if (s < threshold && s > next)
                        {
                            next = s;
                        }
                    }

                    threshold = next;
                }
            }

            float sum = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                float s = z[j] * invT;
                e[j] = s >= threshold ? MathF.Exp(s - max) : 0f;
                sum += e[j];
            }

            float target = CounterRandom.Uniform(seed, stepNumber, (uint)r) * sum;
            int chosen = -1;
            float cumulative = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                if (e[j] > 0f)
                {
                    chosen = j;
                    cumulative += e[j];
                    if (cumulative > target)
                    {
                        break;
                    }
                }
            }

            float entropy = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                if (e[j] > 0f)
                {
                    float p = e[j] / sum;
                    entropy -= p * MathF.Log2(p);
                }
            }

            iv[r] = chosen;
            int o = ((int)stepNumber * rows + r) * 13;
            sv[o] = chosen;
            sv[o + 1] = e[chosen] / sum;
            sv[o + 2] = entropy;
            for (int a = 0; a < 5; a++)
            {
                int best = -1;
                for (int j = 0; j < vocabulary; j++)
                {
                    if (e[j] > 0f && (best < 0 || e[j] > e[best]) && taken[..a].IndexOf(j) < 0)
                    {
                        best = j;
                    }
                }

                taken[a] = best;
                sv[o + 3 + 2 * a] = best;
                sv[o + 4 + 2 * a] = best < 0 ? 0f : e[best] / sum;
            }
        }
    }
}
