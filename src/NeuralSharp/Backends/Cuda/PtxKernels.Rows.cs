using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX building blocks for kernels that give each row a block of threads (row reductions over thousands of columns,
// such as a softmax over a 150k-token vocabulary), and matrix products with few rows (token-by-token decoding).
internal static partial class PtxKernels
{
    public static readonly string[] RowNames = ["gemv_nn_f32", "gemv_nt_f32", "attention_decode_f32"];

    /// <summary>Threads per row in <see cref="RowBlock"/> kernels (a multiple of 32, at most 1024).</summary>
    public const int RowThreads = 256;

    /// <summary>Threads of the sampler's block (one row: a whole vocabulary per block, so more loads in flight).</summary>
    public const int SamplerThreads = 1024;

    /// <summary>Rows at most this many take the few-row matrix product kernels.</summary>
    public const int GemvRows = 8;

    /// <summary>
    /// A kernel with one block of <see cref="RowThreads"/> threads per row (row = block index, below the last
    /// parameter n). Parameters are laid out as in <see cref="Elementwise"/>, so launches pass the same arguments.
    /// The body sees %row (also in %i), %tx (thread index), %nt (threads per block), %lane, %warp, %nwarps, %b_x buffer bases,
    /// %a_x = %b_x + 4·row, %s_x scalars; %r/%f/%rd/%p registers are free except those the macros below document.
    /// </summary>
    private static void RowBlock(StringBuilder sb, string name, string[] pointers, (string Type, string Name)[] scalars, string body,
        int sharedFloats = 0)
    {
        var parameters = pointers.Select(p => $"    .param .u64 p_{p}")
            .Concat(scalars.Select(s => $"    .param .{s.Type} p_{s.Name}"))
            .Append("    .param .u32 p_n");

        sb.AppendLine($".visible .entry {name}(");
        sb.AppendLine(string.Join(",\n", parameters));
        sb.AppendLine(")");
        sb.AppendLine("{");
        sb.AppendLine("    .reg .pred %p<16>;");
        sb.AppendLine("    .reg .f32 %f<32>;");
        sb.AppendLine("    .reg .b32 %r<32>;");
        sb.AppendLine("    .reg .b64 %rd<16>;");
        sb.AppendLine("    .reg .u32 %i, %n, %row, %tx, %nt, %lane, %warp, %nwarps, %sb, %sbi, %spart;");
        sb.AppendLine("    .reg .u64 %off;");
        sb.AppendLine("    .reg .f32 %bf<4>;");
        sb.AppendLine("    .reg .b32 %br<6>;");
        sb.AppendLine("    .reg .pred %bp<4>;");
        sb.AppendLine($"    .shared .align 4 .b32 {name}_rv[32];");
        sb.AppendLine($"    .shared .align 4 .b32 {name}_ri[32];");
        if (sharedFloats > 0)
        {
            sb.AppendLine($"    .shared .align 4 .b32 {name}_part[{sharedFloats}];");
        }

        foreach (var p in pointers)
        {
            sb.AppendLine($"    .reg .u64 %a_{p}, %b_{p};");
        }

        foreach (var s in scalars)
        {
            sb.AppendLine($"    .reg .{s.Type} %s_{s.Name};");
        }

        sb.AppendLine($"""
                mov.u32 %row, %ctaid.x;
                ld.param.u32 %n, [p_n];
                setp.ge.u32 %p0, %row, %n;
                @%p0 bra DONE;
                mov.u32 %i, %row;
                mov.u32 %tx, %tid.x;
                mov.u32 %nt, %ntid.x;
                and.b32 %lane, %tx, 31;
                shr.u32 %warp, %tx, 5;
                add.u32 %nwarps, %nt, 31;
                shr.u32 %nwarps, %nwarps, 5;
                mov.u32 %sb, {name}_rv;
                mov.u32 %sbi, {name}_ri;
                mul.wide.u32 %off, %row, 4;
            """);
        if (sharedFloats > 0)
        {
            sb.AppendLine($"    mov.u32 %spart, {name}_part;");
        }

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

    /// <summary>
    /// Loop of this thread over columns tid, tid + nt, … below <paramref name="limit"/> of the row at
    /// <paramref name="rowPtr"/>: %r6 = column, %rd5 = its address, %f2 = its value. "bra {label}_NEXT" continues.
    /// </summary>
    private static string StridedLoop(string label, string rowPtr, string limit, string body) => $"""
        mov.u32 %r6, %tx;
        {label}:
        setp.ge.u32 %p1, %r6, {limit};
        @%p1 bra {label}_END;
        mul.wide.u32 %rd5, %r6, 4;
        add.u64 %rd5, %rd5, {rowPtr};
        ld.global.f32 %f2, [%rd5];
        {body}
        {label}_NEXT:
        add.u32 %r6, %r6, %nt;
        bra {label};
        {label}_END:
        """;

    /// <summary>
    /// Combines <paramref name="reg"/> over the block with add, max or min; every thread ends with the result. All
    /// threads of the block must reach it. Uses %bf*, %br*, %bp* and the block's reduction slots.
    /// </summary>
    private static string BlockReduce(string label, string reg, string op, string identity)
    {
        var s = new StringBuilder();
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine($"shfl.sync.bfly.b32 %bf0, {reg}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"{op}.f32 {reg}, {reg}, %bf0;");
        }

        s.Append($"""
            setp.eq.u32 %bp0, %lane, 0;
            shl.b32 %br0, %warp, 2;
            add.u32 %br0, %br0, %sb;
            @%bp0 st.shared.f32 [%br0], {reg};
            bar.sync 0;
            mov.f32 {reg}, {identity};
            mov.u32 %br1, 0;
            {label}:
            setp.ge.u32 %bp1, %br1, %nwarps;
            @%bp1 bra {label}_END;
            shl.b32 %br0, %br1, 2;
            add.u32 %br0, %br0, %sb;
            ld.shared.f32 %bf0, [%br0];
            {op}.f32 {reg}, {reg}, %bf0;
            add.u32 %br1, %br1, 1;
            bra {label};
            {label}_END:
            bar.sync 0;
            """);
        return s.ToString();
    }

    /// <summary>
    /// Arg-max over the block of (<paramref name="value"/>, <paramref name="index"/>) pairs: the larger value wins,
    /// on ties the smaller index (as unsigned, so -1 loses). Every thread ends with the winner.
    /// </summary>
    private static string BlockArgMax(string label, string value, string index)
    {
        const string Better = """
            setp.gt.f32 %bp0, %bf0, {0};
            setp.eq.f32 %bp1, %bf0, {0};
            setp.lt.u32 %bp2, %br2, {1};
            and.pred %bp1, %bp1, %bp2;
            or.pred %bp0, %bp0, %bp1;
            selp.f32 {0}, %bf0, {0}, %bp0;
            selp.b32 {1}, %br2, {1}, %bp0;
            """;
        var s = new StringBuilder();
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            s.AppendLine($"shfl.sync.bfly.b32 %bf0, {value}, {offset}, 31, 0xffffffff;");
            s.AppendLine($"shfl.sync.bfly.b32 %br2, {index}, {offset}, 31, 0xffffffff;");
            s.AppendLine(string.Format(Better, value, index));
        }

        s.Append($"""
            setp.eq.u32 %bp3, %lane, 0;
            shl.b32 %br0, %warp, 2;
            add.u32 %br3, %br0, %sb;
            @%bp3 st.shared.f32 [%br3], {value};
            add.u32 %br3, %br0, %sbi;
            @%bp3 st.shared.b32 [%br3], {index};
            bar.sync 0;
            ld.shared.f32 {value}, [%sb];
            ld.shared.b32 {index}, [%sbi];
            mov.u32 %br1, 1;
            {label}:
            setp.ge.u32 %bp3, %br1, %nwarps;
            @%bp3 bra {label}_END;
            shl.b32 %br0, %br1, 2;
            add.u32 %br3, %br0, %sb;
            ld.shared.f32 %bf0, [%br3];
            add.u32 %br3, %br0, %sbi;
            ld.shared.b32 %br2, [%br3];
            """ + "\n" + string.Format(Better, value, index) + "\n" + $"""
            add.u32 %br1, %br1, 1;
            bra {label};
            {label}_END:
            bar.sync 0;
            """);
        return s.ToString();
    }

    /// <summary>Largest head size <c>attention_decode_f32</c> handles (8 dimensions per lane).</summary>
    public const int DecodeMaxDim = 256;

    private static void BuildRows(StringBuilder sb)
    {
        GemvNN(sb);
        GemvNT(sb);
        AttentionDecode(sb);
    }

    // Attention of one query row over the filled part of a key/value cache (see Backend.AttentionDecode). One block
    // per row; warp w takes positions w, w + 8, … with the lanes splitting the head dimension (lane + 32·i), keeping a
    // running maximum, softmax sum and weighted value sum (online softmax). The warps' partial results are then
    // combined in shared memory: part[w] = (max, sum, acc[256]).
    private static void AttentionDecode(StringBuilder sb)
    {
        const int Stride = (2 + DecodeMaxDim) * 4;
        var b = new StringBuilder();
        b.AppendLine("""
            div.u32 %r7, %row, %s_rph;
            rem.u32 %r8, %row, %s_rph;
            rem.u32 %r9, %r8, %s_steps;
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r10, %f1;
            add.u32 %r10, %r10, %r9;
            sub.u32 %r11, %s_cap, 1;
            min.u32 %r10, %r10, %r11;
            add.u32 %r10, %r10, 1;
            mul.lo.u32 %r12, %row, %s_dim;
            mul.wide.u32 %rd1, %r12, 4;
            add.u64 %rd2, %b_q, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            mul.lo.u32 %r13, %r7, %s_cap;
            mul.wide.u32 %rd4, %r13, %s_dim;
            shl.b64 %rd4, %rd4, 2;
            add.u64 %rd5, %b_keys, %rd4;
            add.u64 %rd6, %b_values, %rd4;
            mul.wide.u32 %rd8, %lane, 4;
            add.u64 %rd9, %rd2, %rd8;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"""
                add.u32 %r14, %lane, {32 * i};
                setp.lt.u32 %p{1 + i}, %r14, %s_dim;
                mov.f32 %f{10 + i}, {Zero};
                @%p{1 + i} ld.global.f32 %f{10 + i}, [%rd9+{128 * i}];
                mov.f32 %f{22 + i}, {Zero};
                """);
        }

        b.AppendLine($"""
            mov.f32 %f20, {NegInf};
            mov.f32 %f21, {Zero};
            mov.u32 %r15, %warp;
            DL:
            setp.ge.u32 %p9, %r15, %r10;
            @%p9 bra DL_END;
            mul.wide.u32 %rd10, %r15, %s_dim;
            shl.b64 %rd10, %rd10, 2;
            add.u64 %rd10, %rd10, %rd8;
            add.u64 %rd11, %rd5, %rd10;
            add.u64 %rd12, %rd6, %rd10;
            mov.f32 %f2, {Zero};
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"@%p{1 + i} ld.global.f32 %f3, [%rd11+{128 * i}];");
            b.AppendLine($"@%p{1 + i} fma.rn.f32 %f2, %f{10 + i}, %f3, %f2;");
        }

        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            b.AppendLine($"shfl.sync.bfly.b32 %f3, %f2, {offset}, 31, 0xffffffff;");
            b.AppendLine("add.f32 %f2, %f2, %f3;");
        }

        b.AppendLine($"""
            mul.f32 %f2, %f2, %s_scale;
            max.f32 %f4, %f20, %f2;
            sub.f32 %f5, %f20, %f4;
            mul.f32 %f5, %f5, {Log2E};
            ex2.approx.ftz.f32 %f5, %f5;
            sub.f32 %f6, %f2, %f4;
            mul.f32 %f6, %f6, {Log2E};
            ex2.approx.ftz.f32 %f6, %f6;
            fma.rn.f32 %f21, %f21, %f5, %f6;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"mul.f32 %f{22 + i}, %f{22 + i}, %f5;");
            b.AppendLine($"@%p{1 + i} ld.global.f32 %f3, [%rd12+{128 * i}];");
            b.AppendLine($"@%p{1 + i} fma.rn.f32 %f{22 + i}, %f6, %f3, %f{22 + i};");
        }

        b.AppendLine($"""
            mov.f32 %f20, %f4;
            add.u32 %r15, %r15, %nwarps;
            bra DL;
            DL_END:
            mul.lo.u32 %r16, %warp, {Stride};
            add.u32 %r16, %r16, %spart;
            setp.eq.u32 %p10, %lane, 0;
            @%p10 st.shared.f32 [%r16], %f20;
            @%p10 st.shared.f32 [%r16+4], %f21;
            shl.b32 %r17, %lane, 2;
            add.u32 %r17, %r17, %r16;
            """);
        for (int i = 0; i < 8; i++)
        {
            b.AppendLine($"@%p{1 + i} st.shared.f32 [%r17+{8 + 128 * i}], %f{22 + i};");
        }

        b.AppendLine($"""
            bar.sync 0;
            mov.f32 %f7, {NegInf};
            mov.u32 %r18, 0;
            CM:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra CM_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            max.f32 %f7, %f7, %f8;
            add.u32 %r18, %r18, 1;
            bra CM;
            CM_END:
            mov.f32 %f9, {Zero};
            mov.u32 %r18, 0;
            CL:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra CL_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            sub.f32 %f8, %f8, %f7;
            mul.f32 %f8, %f8, {Log2E};
            ex2.approx.ftz.f32 %f8, %f8;
            ld.shared.f32 %f31, [%r19+4];
            fma.rn.f32 %f9, %f31, %f8, %f9;
            add.u32 %r18, %r18, 1;
            bra CL;
            CL_END:
            rcp.rn.f32 %f9, %f9;
            mov.u32 %r20, %tx;
            CO:
            setp.ge.u32 %p12, %r20, %s_dim;
            @%p12 bra CO_END;
            mov.f32 %f30, {Zero};
            mov.u32 %r18, 0;
            COW:
            setp.ge.u32 %p11, %r18, %nwarps;
            @%p11 bra COW_END;
            mul.lo.u32 %r19, %r18, {Stride};
            add.u32 %r19, %r19, %spart;
            ld.shared.f32 %f8, [%r19];
            sub.f32 %f8, %f8, %f7;
            mul.f32 %f8, %f8, {Log2E};
            ex2.approx.ftz.f32 %f8, %f8;
            shl.b32 %r21, %r20, 2;
            add.u32 %r21, %r21, %r19;
            ld.shared.f32 %f31, [%r21+8];
            fma.rn.f32 %f30, %f31, %f8, %f30;
            add.u32 %r18, %r18, 1;
            bra COW;
            COW_END:
            mul.f32 %f30, %f30, %f9;
            mul.wide.u32 %rd13, %r20, 4;
            add.u64 %rd13, %rd13, %rd3;
            st.global.f32 [%rd13], %f30;
            add.u32 %r20, %r20, %nt;
            bra CO;
            CO_END:
            """);
        RowBlock(sb, "attention_decode_f32", ["q", "keys", "values", "pos", "y"],
            [("u32", "rph"), ("u32", "steps"), ("u32", "cap"), ("u32", "dim"), ("f32", "scale")], b.ToString(),
            sharedFloats: RowThreads / 32 * (2 + DecodeMaxDim));
    }

    /// <summary>Threads of a <c>gemv_nn_f32</c> block: 32 columns × 32 slices of k.</summary>
    public const int GemvThreads = 1024;

    // c[b][r, j] = Σ_k a[b][r, k] · w[b][k, j] (+ beta · c) for m ≤ 8 rows. Block: 32 columns × 32 slices of k (slice s
    // takes k = s, s + 32, …, four at a time so several loads are in flight); each thread keeps 8 row sums, the slices
    // are added in shared memory and warp r writes row r. Grid: x = ⌈n / 32⌉, z = batch. Reads of w are coalesced.
    private static void GemvNN(StringBuilder sb)
    {
        var s = new StringBuilder();
        s.AppendLine("""
            .visible .entry gemv_nn_f32(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc
            )
            {
                .reg .pred %p<16>;
                .reg .f32 %f<32>;
                .reg .b32 %r<24>;
                .reg .b64 %rd<24>;
                .shared .align 4 .f32 gemv_nn_part[8192];
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.f32 %f20, [p_beta];
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
                mov.u32 %r4, %ctaid.z;
                cvt.u64.u32 %rd7, %r4;
                mul.lo.u64 %rd8, %rd7, %rd4;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd1, %rd1, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd5;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd2, %rd2, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd6;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd3, %rd3, %rd8;
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 31;
                shr.u32 %r7, %r5, 5;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 5;
                add.u32 %r8, %r8, %r6;
                setp.lt.u32 %p9, %r8, %r2;
                mul.wide.u32 %rd12, %r3, 4;
                mul.wide.u32 %rd14, %r2, 128;
                cvt.u64.u32 %rd10, %r8;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    mov.f32 %f{r}, 0f00000000;");
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
        }

        // Four k at a time: kk, kk + 32, kk + 64, kk + 96 while kk + 96 < k.
        s.AppendLine("""
                mov.u32 %r9, %r7;
            K4:
                add.u32 %r14, %r9, 96;
                setp.ge.u32 %p10, %r14, %r3;
                @%p10 bra K1;
                mul.wide.u32 %rd9, %r9, %r2;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                mov.f32 %f8, 0f00000000;
                mov.f32 %f10, 0f00000000;
                mov.f32 %f11, 0f00000000;
                mov.f32 %f12, 0f00000000;
                @%p9 ld.global.f32 %f8, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f10, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f11, [%rd9];
                add.u64 %rd9, %rd9, %rd14;
                @%p9 ld.global.f32 %f12, [%rd9];
                mul.wide.u32 %rd11, %r9, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"""
                    @!%p{r} bra K4_ROWS_DONE;
                    ld.global.f32 %f13, [%rd11];
                    ld.global.f32 %f14, [%rd11+128];
                    ld.global.f32 %f15, [%rd11+256];
                    ld.global.f32 %f16, [%rd11+384];
                    fma.rn.f32 %f{r}, %f13, %f8, %f{r};
                    fma.rn.f32 %f{r}, %f14, %f10, %f{r};
                    fma.rn.f32 %f{r}, %f15, %f11, %f{r};
                    fma.rn.f32 %f{r}, %f16, %f12, %f{r};
                    add.u64 %rd11, %rd11, %rd12;
                """);
        }

        s.AppendLine("""
            K4_ROWS_DONE:
                add.u32 %r9, %r9, 128;
                bra K4;
            K1:
                setp.ge.u32 %p10, %r9, %r3;
                @%p10 bra KEND;
                mov.f32 %f8, 0f00000000;
                mul.wide.u32 %rd9, %r9, %r2;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                @%p9 ld.global.f32 %f8, [%rd9];
                mul.wide.u32 %rd11, %r9, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    @%p{r} ld.global.f32 %f9, [%rd11];");
            s.AppendLine($"    @%p{r} fma.rn.f32 %f{r}, %f9, %f8, %f{r};");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        }

        // part[slice][row][lane]: slice stride 1024 bytes, row stride 128 bytes.
        s.AppendLine("""
                add.u32 %r9, %r9, 32;
                bra K1;
            KEND:
                mov.u32 %r10, gemv_nn_part;
                shl.b32 %r11, %r7, 10;
                shl.b32 %r12, %r6, 2;
                add.u32 %r11, %r11, %r12;
                add.u32 %r11, %r11, %r10;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    st.shared.f32 [%r11+{r * 128}], %f{r};");
        }

        s.AppendLine("""
                bar.sync 0;
                setp.ge.u32 %p11, %r7, %r1;
                @%p11 bra DONE;
                @!%p9 bra DONE;
                shl.b32 %r13, %r7, 7;
                add.u32 %r13, %r13, %r12;
                add.u32 %r13, %r13, %r10;
                mov.f32 %f17, 0f00000000;
            """);
        for (int slice = 0; slice < 32; slice++)
        {
            s.AppendLine($"    ld.shared.f32 %f18, [%r13+{slice * 1024}];");
            s.AppendLine("    add.f32 %f17, %f17, %f18;");
        }

        s.AppendLine("""
                mul.wide.u32 %rd13, %r7, %r2;
                add.u64 %rd13, %rd13, %rd10;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd3;
                setp.eq.f32 %p12, %f20, 0f00000000;
                @%p12 bra STORE;
                ld.global.f32 %f19, [%rd13];
                fma.rn.f32 %f17, %f19, %f20, %f17;
            STORE:
                st.global.f32 [%rd13], %f17;
            DONE:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }

    // c[b][r, j] = Σ_k a[b][r, k] · w[b][j, k] (+ beta · c), w given as [n, k] (a product with a transposed matrix,
    // such as queries times keys), for m ≤ 8 rows. One warp per column: lanes stride over k (coalesced), then a warp
    // reduction per row. Grid: x = ⌈n / 8⌉ (8 warps per block), z = batch.
    private static void GemvNT(StringBuilder sb)
    {
        var s = new StringBuilder();
        s.AppendLine("""
            .visible .entry gemv_nt_f32(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc
            )
            {
                .reg .pred %p<16>;
                .reg .f32 %f<24>;
                .reg .b32 %r<24>;
                .reg .b64 %rd<24>;
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.f32 %f20, [p_beta];
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
                mov.u32 %r4, %ctaid.z;
                cvt.u64.u32 %rd7, %r4;
                mul.lo.u64 %rd8, %rd7, %rd4;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd1, %rd1, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd5;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd2, %rd2, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd6;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd3, %rd3, %rd8;
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 31;
                shr.u32 %r7, %r5, 5;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 3;
                add.u32 %r8, %r8, %r7;
                setp.ge.u32 %p9, %r8, %r2;
                @%p9 bra DONE;
                mul.wide.u32 %rd9, %r8, %r3;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    mov.f32 %f{r}, 0f00000000;");
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
        }

        s.AppendLine("""
                mov.u32 %r9, %r6;
                mul.wide.u32 %rd12, %r3, 4;
            KLOOP:
                setp.ge.u32 %p10, %r9, %r3;
                @%p10 bra KEND;
                mul.wide.u32 %rd10, %r9, 4;
                add.u64 %rd11, %rd10, %rd9;
                ld.global.f32 %f8, [%rd11];
                add.u64 %rd11, %rd10, %rd1;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    @%p{r} ld.global.f32 %f9, [%rd11];");
            s.AppendLine($"    @%p{r} fma.rn.f32 %f{r}, %f9, %f8, %f{r};");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        }

        s.AppendLine("""
                add.u32 %r9, %r9, 32;
                bra KLOOP;
            KEND:
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            foreach (int offset in new[] { 16, 8, 4, 2, 1 })
            {
                s.AppendLine($"    shfl.sync.bfly.b32 %f9, %f{r}, {offset}, 31, 0xffffffff;");
                s.AppendLine($"    add.f32 %f{r}, %f{r}, %f9;");
            }
        }

        s.AppendLine("""
                setp.ne.u32 %p11, %r6, 0;
                @%p11 bra DONE;
                setp.ne.f32 %p12, %f20, 0f00000000;
                cvt.u64.u32 %rd13, %r8;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd3;
                mul.wide.u32 %rd14, %r2, 4;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"""
                    @!%p{r} bra DONE;
                    @%p12 ld.global.f32 %f10, [%rd13];
                    @%p12 fma.rn.f32 %f{r}, %f10, %f20, %f{r};
                    st.global.f32 [%rd13], %f{r};
                    add.u64 %rd13, %rd13, %rd14;
                """);
        }

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        sb.AppendLine(s.ToString());
    }
}
