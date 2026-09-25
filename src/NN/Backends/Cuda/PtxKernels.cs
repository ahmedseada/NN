using System.Globalization;
using System.Text;

namespace NN.Backends.Cuda;

/// <summary>
/// Generates the PTX (NVIDIA's virtual GPU assembly) for every kernel the CUDA backend uses.
/// The driver JIT-compiles this text for the installed GPU when the module is loaded, so the
/// library needs neither nvcc nor NVRTC nor any precompiled GPU binaries.
/// </summary>
internal static class PtxKernels
{
    /// <summary>Threads per block for 1-D kernels. The reduction in <c>sum_f32</c> is unrolled for exactly this size.</summary>
    public const int BlockSize = 256;

    /// <summary>Tile edge of the shared-memory matrix multiply; blocks are Tile x Tile threads.</summary>
    public const int Tile = 16;

    private const string One = "0f3F800000";
    private const string Zero = "0f00000000";

    public static readonly string[] Names =
    [
        "fill_f32", "affine_f32", "axpy_f32", "muladd_f32",
        "add_f32", "sub_f32", "mul_f32",
        "sigmoid_f32", "tanh_f32", "relu_f32", "square_f32",
        "sigmoid_bwd_f32", "tanh_bwd_f32", "relu_bwd_f32", "square_bwd_f32",
        "add_rowvec_f32", "add_scalar_f32", "sum_rows_f32", "sum_f32",
        "sgd_momentum_f32", "adam_f32", "matmul_f32",
    ];

    public static string Source { get; } = Build();

    private static string Build()
    {
        var sb = new StringBuilder();
        sb.AppendLine(".version 6.0");
        sb.AppendLine(".target sm_50");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();

        Elementwise(sb, "fill_f32", ["y"], [("f32", "value")],
            "st.global.f32 [%a_y], %s_value;");

        Elementwise(sb, "affine_f32", ["x", "y"], [("f32", "alpha"), ("f32", "beta")],
            """
            ld.global.f32 %f1, [%a_x];
            fma.rn.f32 %f2, %f1, %s_alpha, %s_beta;
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "axpy_f32", ["x", "y"], [("f32", "alpha")],
            """
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%a_y];
            fma.rn.f32 %f2, %f1, %s_alpha, %f2;
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "muladd_f32", ["a", "b", "c"], [],
            """
            ld.global.f32 %f1, [%a_a];
            ld.global.f32 %f2, [%a_b];
            ld.global.f32 %f3, [%a_c];
            fma.rn.f32 %f3, %f1, %f2, %f3;
            st.global.f32 [%a_c], %f3;
            """);

        foreach (var (name, instr) in new[] { ("add_f32", "add.f32"), ("sub_f32", "sub.f32"), ("mul_f32", "mul.f32") })
        {
            Elementwise(sb, name, ["a", "b", "c"], [],
                $"""
                ld.global.f32 %f1, [%a_a];
                ld.global.f32 %f2, [%a_b];
                {instr} %f3, %f1, %f2;
                st.global.f32 [%a_c], %f3;
                """);
        }

        // sigmoid(x) = 1 / (1 + 2^(-x * log2(e)))
        Elementwise(sb, "sigmoid_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            mul.f32 %f2, %f1, {F(-1.4426950408889634f)};
            ex2.approx.ftz.f32 %f2, %f2;
            add.f32 %f2, %f2, {One};
            rcp.rn.f32 %f3, %f2;
            st.global.f32 [%a_y], %f3;
            """);

        // tanh(x) = 1 - 2 / (2^(2x * log2(e)) + 1)
        Elementwise(sb, "tanh_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            mul.f32 %f2, %f1, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f2, %f2;
            add.f32 %f2, %f2, {One};
            rcp.rn.f32 %f3, %f2;
            fma.rn.f32 %f4, %f3, {F(-2f)}, {One};
            st.global.f32 [%a_y], %f4;
            """);

        Elementwise(sb, "relu_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            max.f32 %f2, %f1, {Zero};
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "square_f32", ["x", "y"], [],
            """
            ld.global.f32 %f1, [%a_x];
            mul.f32 %f2, %f1, %f1;
            st.global.f32 [%a_y], %f2;
            """);

        // Backward kernels: dx += dy * f'(.)
        Elementwise(sb, "sigmoid_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_y];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            sub.f32 %f4, {One}, %f1;
            mul.f32 %f4, %f4, %f1;
            fma.rn.f32 %f3, %f2, %f4, %f3;
            st.global.f32 [%a_dx], %f3;
            """);

        Elementwise(sb, "tanh_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_y];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            mul.f32 %f4, %f1, %f1;
            sub.f32 %f4, {One}, %f4;
            fma.rn.f32 %f3, %f2, %f4, %f3;
            st.global.f32 [%a_dx], %f3;
            """);

        Elementwise(sb, "relu_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            setp.gt.f32 %p1, %f1, {Zero};
            selp.f32 %f4, %f2, {Zero}, %p1;
            add.f32 %f3, %f3, %f4;
            st.global.f32 [%a_dx], %f3;
            """);

        Elementwise(sb, "square_bwd_f32", ["x", "y", "dy", "dx"], [],
            """
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            add.f32 %f4, %f1, %f1;
            fma.rn.f32 %f3, %f4, %f2, %f3;
            st.global.f32 [%a_dx], %f3;
            """);

        // c[i] = a[i] + v[i % cols]
        Elementwise(sb, "add_rowvec_f32", ["a", "v", "c"], [("u32", "cols")],
            """
            rem.u32 %r5, %i, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_v, %rd1;
            ld.global.f32 %f1, [%a_a];
            ld.global.f32 %f2, [%rd2];
            add.f32 %f3, %f1, %f2;
            st.global.f32 [%a_c], %f3;
            """);

        // y[i] += scale * s[0]
        Elementwise(sb, "add_scalar_f32", ["s", "y"], [("f32", "scale")],
            """
            ld.global.f32 %f1, [%b_s];
            ld.global.f32 %f2, [%a_y];
            fma.rn.f32 %f2, %f1, %s_scale, %f2;
            st.global.f32 [%a_y], %f2;
            """);

        // v = momentum * v + g; p += nlr * v   (nlr = -learning rate)
        Elementwise(sb, "sgd_momentum_f32", ["p", "g", "v"], [("f32", "nlr"), ("f32", "momentum")],
            """
            ld.global.f32 %f1, [%a_p];
            ld.global.f32 %f2, [%a_g];
            ld.global.f32 %f3, [%a_v];
            fma.rn.f32 %f3, %s_momentum, %f3, %f2;
            fma.rn.f32 %f1, %f3, %s_nlr, %f1;
            st.global.f32 [%a_v], %f3;
            st.global.f32 [%a_p], %f1;
            """);

        // m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; p -= lr * m / (sqrt(v) + eps)
        Elementwise(sb, "adam_f32", ["p", "g", "m", "v"],
            [("f32", "lr"), ("f32", "b1"), ("f32", "b2"), ("f32", "c1"), ("f32", "c2"), ("f32", "eps")],
            """
            ld.global.f32 %f1, [%a_p];
            ld.global.f32 %f2, [%a_g];
            ld.global.f32 %f3, [%a_m];
            ld.global.f32 %f4, [%a_v];
            mul.f32 %f5, %f2, %s_c1;
            fma.rn.f32 %f3, %s_b1, %f3, %f5;
            mul.f32 %f6, %f2, %f2;
            mul.f32 %f6, %f6, %s_c2;
            fma.rn.f32 %f4, %s_b2, %f4, %f6;
            sqrt.rn.f32 %f7, %f4;
            add.f32 %f7, %f7, %s_eps;
            div.rn.f32 %f8, %f3, %f7;
            mul.f32 %f8, %f8, %s_lr;
            sub.f32 %f1, %f1, %f8;
            st.global.f32 [%a_m], %f3;
            st.global.f32 [%a_v], %f4;
            st.global.f32 [%a_p], %f1;
            """);

        SumRows(sb);
        Sum(sb);
        MatMul(sb);
        return sb.ToString();
    }

    /// <summary>Formats a float as a PTX hexadecimal literal (bit-exact).</summary>
    private static string F(float value) => "0f" + BitConverter.SingleToUInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    /// <summary>
    /// Emits a one-thread-per-element kernel over [0, n). For each pointer parameter p the body can use
    /// %b_p (global base address) and %a_p (address of element i); scalars are in %s_name; the index is %i.
    /// </summary>
    private static void Elementwise(StringBuilder sb, string name, string[] pointers, (string Type, string Name)[] scalars, string body)
    {
        var parameters = pointers.Select(p => $"    .param .u64 p_{p}")
            .Concat(scalars.Select(s => $"    .param .{s.Type} p_{s.Name}"))
            .Append("    .param .u32 p_n");

        sb.AppendLine($".visible .entry {name}(");
        sb.AppendLine(string.Join(",\n", parameters));
        sb.AppendLine(")");
        sb.AppendLine("{");
        sb.AppendLine("    .reg .pred %p<4>;");
        sb.AppendLine("    .reg .f32 %f<16>;");
        sb.AppendLine("    .reg .b32 %r<8>;");
        sb.AppendLine("    .reg .b64 %rd<4>;");
        sb.AppendLine("    .reg .u32 %i, %n;");
        sb.AppendLine("    .reg .u64 %off;");
        foreach (var p in pointers)
        {
            sb.AppendLine($"    .reg .u64 %a_{p}, %b_{p};");
        }

        foreach (var s in scalars)
        {
            sb.AppendLine($"    .reg .{s.Type} %s_{s.Name};");
        }

        sb.AppendLine("""
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %i, %r1, %r2, %r3;
                ld.param.u32 %n, [p_n];
                setp.ge.u32 %p0, %i, %n;
                @%p0 bra DONE;
                mul.wide.u32 %off, %i, 4;
            """);
        foreach (var p in pointers)
        {
            sb.AppendLine($"    ld.param.u64 %b_{p}, [p_{p}];");
            sb.AppendLine($"    cvta.to.global.u64 %b_{p}, %b_{p};");
            sb.AppendLine($"    add.u64 %a_{p}, %b_{p}, %off;");
        }

        foreach (var s in scalars)
        {
            sb.AppendLine($"    ld.param.{s.Type} %s_{s.Name}, [p_{s.Name}];");
        }

        foreach (var line in body.Split('\n'))
        {
            sb.Append("    ").AppendLine(line.TrimEnd());
        }

        sb.AppendLine("DONE:");
        sb.AppendLine("    ret;");
        sb.AppendLine("}");
        sb.AppendLine();
    }

    /// <summary>y[j] += sum over rows of x[r, j]; one thread per column so reads are coalesced.</summary>
    private static void SumRows(StringBuilder sb) => sb.AppendLine($$"""
        .visible .entry sum_rows_f32(
            .param .u64 p_x,
            .param .u64 p_y,
            .param .u32 p_rows,
            .param .u32 p_cols
        )
        {
            .reg .pred %p<4>;
            .reg .f32 %f<4>;
            .reg .b32 %r<8>;
            .reg .b64 %rd<8>;
            mov.u32 %r1, %ctaid.x;
            mov.u32 %r2, %ntid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r4, %r1, %r2, %r3;
            ld.param.u32 %r5, [p_cols];
            setp.ge.u32 %p0, %r4, %r5;
            @%p0 bra DONE;
            ld.param.u32 %r6, [p_rows];
            ld.param.u64 %rd1, [p_x];
            cvta.to.global.u64 %rd1, %rd1;
            mul.wide.u32 %rd2, %r4, 4;
            add.u64 %rd1, %rd1, %rd2;
            mul.wide.u32 %rd3, %r5, 4;
            mov.f32 %f1, {{Zero}};
            mov.u32 %r7, 0;
        LOOP:
            setp.ge.u32 %p1, %r7, %r6;
            @%p1 bra END;
            ld.global.f32 %f2, [%rd1];
            add.f32 %f1, %f1, %f2;
            add.u64 %rd1, %rd1, %rd3;
            add.u32 %r7, %r7, 1;
            bra LOOP;
        END:
            ld.param.u64 %rd4, [p_y];
            cvta.to.global.u64 %rd4, %rd4;
            add.u64 %rd4, %rd4, %rd2;
            ld.global.f32 %f3, [%rd4];
            add.f32 %f3, %f3, %f1;
            st.global.f32 [%rd4], %f3;
        DONE:
            ret;
        }

        """);

    /// <summary>
    /// out[0] += scale * sum(x). Grid-stride accumulation per thread, a shared-memory tree reduction
    /// per block, then one atomic add per block. The caller zeroes out[0] first.
    /// </summary>
    private static void Sum(StringBuilder sb)
    {
        sb.AppendLine($$"""
            .visible .entry sum_f32(
                .param .u64 p_x,
                .param .u64 p_out,
                .param .u32 p_n,
                .param .f32 p_scale
            )
            {
                .reg .pred %p<4>;
                .reg .f32 %f<8>;
                .reg .b32 %r<12>;
                .reg .b64 %rd<8>;
                .shared .align 4 .f32 sdata[{{BlockSize}}];
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ntid.x;
                mov.u32 %r3, %tid.x;
                mad.lo.u32 %r4, %r1, %r2, %r3;
                mov.u32 %r5, %nctaid.x;
                mul.lo.u32 %r5, %r5, %r2;
                ld.param.u32 %r6, [p_n];
                ld.param.u64 %rd1, [p_x];
                cvta.to.global.u64 %rd1, %rd1;
                mov.f32 %f1, {{Zero}};
            LOOP:
                setp.ge.u32 %p1, %r4, %r6;
                @%p1 bra REDUCE;
                mul.wide.u32 %rd2, %r4, 4;
                add.u64 %rd3, %rd1, %rd2;
                ld.global.f32 %f2, [%rd3];
                add.f32 %f1, %f1, %f2;
                add.u32 %r4, %r4, %r5;
                bra LOOP;
            REDUCE:
                mov.u32 %r7, sdata;
                shl.b32 %r8, %r3, 2;
                add.u32 %r9, %r7, %r8;
                st.shared.f32 [%r9], %f1;
                bar.sync 0;
            """);

        for (int stride = BlockSize / 2; stride > 0; stride /= 2)
        {
            sb.AppendLine($$"""
                    setp.ge.u32 %p2, %r3, {{stride}};
                    @%p2 bra SKIP{{stride}};
                    ld.shared.f32 %f3, [%r9];
                    ld.shared.f32 %f4, [%r9+{{stride * 4}}];
                    add.f32 %f3, %f3, %f4;
                    st.shared.f32 [%r9], %f3;
                SKIP{{stride}}:
                    bar.sync 0;
                """);
        }

        sb.AppendLine("""
                setp.ne.u32 %p3, %r3, 0;
                @%p3 bra DONE;
                ld.shared.f32 %f5, [%r7];
                ld.param.f32 %f6, [p_scale];
                mul.f32 %f5, %f5, %f6;
                ld.param.u64 %rd4, [p_out];
                cvta.to.global.u64 %rd4, %rd4;
                atom.global.add.f32 %f7, [%rd4], %f5;
            DONE:
                ret;
            }

            """);
    }

    /// <summary>
    /// C = op(A) op(B) + beta C with 16x16 shared-memory tiles. Thread (tx, ty) of block (bx, by)
    /// owns C[by*16+ty, bx*16+tx]. Out-of-range tile elements are loaded as zero, so any
    /// m, n, k work. transA / transB select the [k, m] / [n, k] storage layouts.
    /// </summary>
    private static void MatMul(StringBuilder sb)
    {
        sb.AppendLine($$"""
            .visible .entry matmul_f32(
                .param .u64 p_a,
                .param .u64 p_b,
                .param .u64 p_c,
                .param .u32 p_m,
                .param .u32 p_n,
                .param .u32 p_k,
                .param .u32 p_ta,
                .param .u32 p_tb,
                .param .f32 p_beta
            )
            {
                .reg .pred %p<8>;
                .reg .f32 %f<8>;
                .reg .f32 %acc;
                .reg .b32 %r<24>;
                .reg .b64 %rd<12>;
                .reg .u32 %tx, %ty, %row, %col, %m, %n, %k, %t, %sa, %sb, %sarow, %sbcol;
                .reg .pred %pta, %ptb;
                .shared .align 4 .f32 As[{{Tile * Tile}}];
                .shared .align 4 .f32 Bs[{{Tile * Tile}}];

                mov.u32 %tx, %tid.x;
                mov.u32 %ty, %tid.y;
                mov.u32 %r1, %ctaid.x;
                mov.u32 %r2, %ctaid.y;
                mad.lo.u32 %col, %r1, {{Tile}}, %tx;
                mad.lo.u32 %row, %r2, {{Tile}}, %ty;
                ld.param.u32 %m, [p_m];
                ld.param.u32 %n, [p_n];
                ld.param.u32 %k, [p_k];
                ld.param.u32 %r3, [p_ta];
                setp.ne.u32 %pta, %r3, 0;
                ld.param.u32 %r4, [p_tb];
                setp.ne.u32 %ptb, %r4, 0;
                ld.param.u64 %rd1, [p_a];
                cvta.to.global.u64 %rd1, %rd1;
                ld.param.u64 %rd2, [p_b];
                cvta.to.global.u64 %rd2, %rd2;

                // Shared addresses: this thread's slot in As/Bs, the start of its As row and its Bs column.
                mov.u32 %r5, As;
                mov.u32 %r6, Bs;
                mad.lo.u32 %r7, %ty, {{Tile}}, %tx;
                shl.b32 %r7, %r7, 2;
                add.u32 %sa, %r5, %r7;
                add.u32 %sb, %r6, %r7;
                mul.lo.u32 %r8, %ty, {{Tile * 4}};
                add.u32 %sarow, %r5, %r8;
                shl.b32 %r9, %tx, 2;
                add.u32 %sbcol, %r6, %r9;

                mov.f32 %acc, {{Zero}};
                mov.u32 %t, 0;
            TILE:
                setp.ge.u32 %p1, %t, %k;
                @%p1 bra STORE;

                // As[ty][tx] = op(A)[row, t + tx]
                add.u32 %r10, %t, %tx;
                mov.f32 %f1, {{Zero}};
                setp.lt.u32 %p2, %row, %m;
                setp.lt.and.u32 %p2, %r10, %k, %p2;
                @!%p2 bra ASTORE;
                mad.lo.u32 %r11, %row, %k, %r10;
                mad.lo.u32 %r12, %r10, %m, %row;
                selp.u32 %r13, %r12, %r11, %pta;
                mul.wide.u32 %rd3, %r13, 4;
                add.u64 %rd4, %rd1, %rd3;
                ld.global.f32 %f1, [%rd4];
            ASTORE:
                st.shared.f32 [%sa], %f1;

                // Bs[ty][tx] = op(B)[t + ty, col]
                add.u32 %r14, %t, %ty;
                mov.f32 %f2, {{Zero}};
                setp.lt.u32 %p3, %col, %n;
                setp.lt.and.u32 %p3, %r14, %k, %p3;
                @!%p3 bra BSTORE;
                mad.lo.u32 %r15, %r14, %n, %col;
                mad.lo.u32 %r16, %col, %k, %r14;
                selp.u32 %r17, %r16, %r15, %ptb;
                mul.wide.u32 %rd5, %r17, 4;
                add.u64 %rd6, %rd2, %rd5;
                ld.global.f32 %f2, [%rd6];
            BSTORE:
                st.shared.f32 [%sb], %f2;
                bar.sync 0;
            """);

        for (int kk = 0; kk < Tile; kk++)
        {
            sb.AppendLine($"    ld.shared.f32 %f3, [%sarow+{kk * 4}];");
            sb.AppendLine($"    ld.shared.f32 %f4, [%sbcol+{kk * Tile * 4}];");
            sb.AppendLine("    fma.rn.f32 %acc, %f3, %f4, %acc;");
        }

        sb.AppendLine($$"""
                bar.sync 0;
                add.u32 %t, %t, {{Tile}};
                bra TILE;

            STORE:
                setp.lt.u32 %p4, %row, %m;
                setp.lt.and.u32 %p4, %col, %n, %p4;
                @!%p4 bra DONE;
                mad.lo.u32 %r18, %row, %n, %col;
                mul.wide.u32 %rd7, %r18, 4;
                ld.param.u64 %rd8, [p_c];
                cvta.to.global.u64 %rd8, %rd8;
                add.u64 %rd9, %rd8, %rd7;
                ld.param.f32 %f5, [p_beta];
                setp.eq.f32 %p5, %f5, {{Zero}};
                @%p5 bra WRITE;
                ld.global.f32 %f6, [%rd9];
                fma.rn.f32 %acc, %f5, %f6, %acc;
            WRITE:
                st.global.f32 [%rd9], %acc;
            DONE:
                ret;
            }
            """);
    }
}
