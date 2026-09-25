using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralSharp.Backends.Cpu;

/// <summary>
/// Single-precision GEMM: C = op(A) * op(B) + beta * C, all row-major.
/// Rows of C are processed in blocks of <see cref="Mr"/>; each block keeps a
/// <see cref="Mr"/> x (2 * SIMD width) tile of C in registers across the whole k loop,
/// so every B load feeds <see cref="Mr"/> fused multiply-adds.
/// </summary>
internal static class CpuMatMul
{
    private const int Mr = 4;

    /// <summary>m * n * k above which row blocks run in parallel.</summary>
    private const long ParallelWork = 1L << 17;

    public static void Multiply(float[] a, float[] b, float[] c, int m, int n, int k, bool transA, bool transB, float beta)
    {
        float[]? rented = null;
        if (transB)
        {
            // B is stored [n, k]; transpose it once into [k, n] so the kernel always streams contiguous B rows.
            rented = ArrayPool<float>.Shared.Rent(k * n);
            Transpose(b, rented, n, k);
            b = rented;
        }

        // Element (i, p) of op(A) lives at a[i * rowStride + p * colStride].
        nint rowStride = transA ? 1 : k;
        nint colStride = transA ? m : 1;
        int blocks = (m + Mr - 1) / Mr;

        if ((long)m * n * k < ParallelWork || blocks == 1 || !ComputeResources.AllowParallel)
        {
            for (int block = 0; block < blocks; block++)
            {
                RunBlock(a, b, c, block, m, n, k, rowStride, colStride, beta);
            }
        }
        else
        {
            float[] bb = b;
            Parallel.For(0, blocks, ComputeResources.ParallelOptions, block => RunBlock(a, bb, c, block, m, n, k, rowStride, colStride, beta));
        }

        if (rented is not null)
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static void Transpose(float[] src, float[] dst, int rows, int cols)
    {
        const int Tile = 32;
        for (int r0 = 0; r0 < rows; r0 += Tile)
        {
            int r1 = Math.Min(r0 + Tile, rows);
            for (int c0 = 0; c0 < cols; c0 += Tile)
            {
                int c1 = Math.Min(c0 + Tile, cols);
                for (int r = r0; r < r1; r++)
                {
                    for (int col = c0; col < c1; col++)
                    {
                        dst[col * rows + r] = src[r * cols + col];
                    }
                }
            }
        }
    }

    private static void RunBlock(float[] a, float[] b, float[] c, int block, int m, int n, int k, nint rowStride, nint colStride, float beta)
    {
        ref float ra = ref MemoryMarshal.GetArrayDataReference(a);
        ref float rb = ref MemoryMarshal.GetArrayDataReference(b);
        ref float rc = ref MemoryMarshal.GetArrayDataReference(c);
        int i0 = block * Mr;
        int rows = Math.Min(Mr, m - i0);
        int j = 0;
        if (rows == Mr)
        {
            j = Block4(ref ra, ref rb, ref rc, i0, n, k, rowStride, colStride, beta);
        }

        for (int r = 0; r < rows; r++)
        {
            Row(ref ra, ref rb, ref rc, i0 + r, j, n, k, rowStride, colStride, beta);
        }
    }

    /// <summary>Computes the 4-row block for every full 2-vector column tile; returns the first column not yet written.</summary>
    private static int Block4(ref float a, ref float b, ref float c, int i0, int n, int k, nint rowStride, nint colStride, float beta)
    {
        int w = Vector<float>.Count;
        nint a0 = i0 * rowStride, a1 = a0 + rowStride, a2 = a1 + rowStride, a3 = a2 + rowStride;
        int j = 0;
        for (; j + 2 * w <= n; j += 2 * w)
        {
            Vector<float> c00 = default, c01 = default, c10 = default, c11 = default;
            Vector<float> c20 = default, c21 = default, c30 = default, c31 = default;
            ref float bj = ref Unsafe.Add(ref b, j);
            nint ap = 0;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                ref float brow = ref Unsafe.Add(ref bj, (nint)p * n);
                var b0 = Vector.LoadUnsafe(ref brow);
                var b1 = Vector.LoadUnsafe(ref brow, (nuint)w);

                var x = new Vector<float>(Unsafe.Add(ref a, a0 + ap));
                c00 = Vector.FusedMultiplyAdd(x, b0, c00);
                c01 = Vector.FusedMultiplyAdd(x, b1, c01);
                x = new Vector<float>(Unsafe.Add(ref a, a1 + ap));
                c10 = Vector.FusedMultiplyAdd(x, b0, c10);
                c11 = Vector.FusedMultiplyAdd(x, b1, c11);
                x = new Vector<float>(Unsafe.Add(ref a, a2 + ap));
                c20 = Vector.FusedMultiplyAdd(x, b0, c20);
                c21 = Vector.FusedMultiplyAdd(x, b1, c21);
                x = new Vector<float>(Unsafe.Add(ref a, a3 + ap));
                c30 = Vector.FusedMultiplyAdd(x, b0, c30);
                c31 = Vector.FusedMultiplyAdd(x, b1, c31);
            }

            ref float c0 = ref Unsafe.Add(ref c, (nint)i0 * n + j);
            ref float c1 = ref Unsafe.Add(ref c0, n);
            ref float c2 = ref Unsafe.Add(ref c1, n);
            ref float c3 = ref Unsafe.Add(ref c2, n);
            Store(ref c0, c00, beta);
            Store(ref Unsafe.Add(ref c0, w), c01, beta);
            Store(ref c1, c10, beta);
            Store(ref Unsafe.Add(ref c1, w), c11, beta);
            Store(ref c2, c20, beta);
            Store(ref Unsafe.Add(ref c2, w), c21, beta);
            Store(ref c3, c30, beta);
            Store(ref Unsafe.Add(ref c3, w), c31, beta);
        }

        return j;
    }

    /// <summary>Computes row i of C from column j to the end: one vector at a time, then scalars.</summary>
    private static void Row(ref float a, ref float b, ref float c, int i, int j, int n, int k, nint rowStride, nint colStride, float beta)
    {
        int w = Vector<float>.Count;
        nint ai = i * rowStride;
        ref float ci = ref Unsafe.Add(ref c, (nint)i * n);
        for (; j + w <= n; j += w)
        {
            Vector<float> acc = default;
            nint ap = ai;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                acc = Vector.FusedMultiplyAdd(new Vector<float>(Unsafe.Add(ref a, ap)), Vector.LoadUnsafe(ref b, (nuint)((nint)p * n + j)), acc);
            }

            Store(ref Unsafe.Add(ref ci, j), acc, beta);
        }

        for (; j < n; j++)
        {
            float acc = 0f;
            nint ap = ai;
            for (int p = 0; p < k; p++, ap += colStride)
            {
                acc = MathF.FusedMultiplyAdd(Unsafe.Add(ref a, ap), Unsafe.Add(ref b, (nint)p * n + j), acc);
            }

            ref float dst = ref Unsafe.Add(ref ci, j);
            dst = beta == 0f ? acc : acc + beta * dst;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(ref float dst, Vector<float> value, float beta)
    {
        if (beta != 0f)
        {
            value = Vector.FusedMultiplyAdd(new Vector<float>(beta), Vector.LoadUnsafe(ref dst), value);
        }

        value.StoreUnsafe(ref dst);
    }
}
