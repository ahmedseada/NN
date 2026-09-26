using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX building blocks for kernels that give each row a block of threads (row reductions over thousands of columns,
// such as a softmax over a 150k-token vocabulary), and matrix products with few rows (token-by-token decoding).
internal static partial class PtxKernels
{
    public static readonly string[] RowNames = ["gemv_nn_f32", "gemv_nt_f32"];

    /// <summary>Threads per row in <see cref="RowBlock"/> kernels (a multiple of 32, at most 1024).</summary>
    public const int RowThreads = 256;

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

    private static void BuildRows(StringBuilder sb)
    {
        GemvNN(sb);
        GemvNT(sb);
    }

    // c[b][r, j] = Σ_k a[b][r, k] · w[b][k, j] (+ beta · c) for m ≤ 8 rows. Block: 32 columns × 8 slices of k; each
    // thread keeps 8 row sums, the slices are added in shared memory, warp r writes row r.
    // Grid: x = ⌈n / 32⌉, z = batch. Reads of w are coalesced (consecutive columns per warp).
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
                .reg .f32 %f<24>;
                .reg .b32 %r<24>;
                .reg .b64 %rd<24>;
                .shared .align 4 .f32 gemv_nn_part[2048];
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
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    mov.f32 %f{r}, 0f00000000;");
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
        }

        s.AppendLine("""
                mov.u32 %r9, %r7;
            KLOOP:
                setp.ge.u32 %p10, %r9, %r3;
                @%p10 bra KEND;
                mov.f32 %f8, 0f00000000;
                mul.wide.u32 %rd9, %r9, %r2;
                cvt.u64.u32 %rd10, %r8;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                @%p9 ld.global.f32 %f8, [%rd9];
                mul.wide.u32 %rd11, %r9, 4;
                add.u64 %rd11, %rd11, %rd1;
                mul.wide.u32 %rd12, %r3, 4;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    @%p{r} ld.global.f32 %f9, [%rd11];");
            s.AppendLine($"    @%p{r} fma.rn.f32 %f{r}, %f9, %f8, %f{r};");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd12;");
        }

        // part[slice][row][lane]
        s.AppendLine("""
                add.u32 %r9, %r9, 8;
                bra KLOOP;
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
                mov.f32 %f10, 0f00000000;
            """);
        for (int slice = 0; slice < 8; slice++)
        {
            s.AppendLine($"    ld.shared.f32 %f11, [%r13+{slice * 1024}];");
            s.AppendLine("    add.f32 %f10, %f10, %f11;");
        }

        s.AppendLine("""
                mul.wide.u32 %rd13, %r7, %r2;
                cvt.u64.u32 %rd10, %r8;
                add.u64 %rd13, %rd13, %rd10;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd3;
                setp.eq.f32 %p12, %f20, 0f00000000;
                @%p12 bra STORE;
                ld.global.f32 %f12, [%rd13];
                fma.rn.f32 %f10, %f12, %f20, %f10;
            STORE:
                st.global.f32 [%rd13], %f10;
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
