using System.Text;

namespace NeuralSharp.Backends.Cuda;

/// <summary>
/// Tensor-core flash attention (sm_80 and newer), the bfloat16 counterpart of <c>attention_flash_f32</c> and its
/// backward kernels: tiles of q, k, v and dOutput are rounded to bfloat16 as they are staged in shared memory, products
/// run on <c>mma.sync.m16n8k16</c> with float32 sums, and the softmax statistics stay float32 (FlashAttention-2).
/// Head sizes 64 and 128. Layouts as <see cref="Backend.AttentionTiled"/>: q [heads, rowsPerHead, dim], keys and values
/// [heads, capacity, dim]; row i sees positions c ≤ position[0] + (i % steps), c &lt; capacity.
/// </summary>
internal static partial class PtxKernels
{
    /// <summary>Query rows per tensor-core flash block (4 warps × 16), and key positions per tile.</summary>
    public const int FlashTensorRows = 64;

    /// <summary>Query rows per step of <c>flash_tc_bwd_kv</c>.</summary>
    public const int FlashTensorBackwardRows = 32;

    /// <summary>Head sizes the tensor-core flash kernels handle.</summary>
    public static bool FlashTensorDim(int dim) => dim is 64 or 128;

    /// <summary>Dynamic shared memory of the backward kernels (bytes).</summary>
    public static int FlashTensorBackwardQShared(int dim) => 4 * FlashTensorRows * FlashStride(dim);

    /// <inheritdoc cref="FlashTensorBackwardQShared"/>
    public static int FlashTensorBackwardKvShared(int dim) => 2 * FlashTensorRows * FlashStride(dim) + 2 * FlashTensorBackwardRows * FlashStride(dim) + 2 * FlashTensorBackwardRows * 4;

    private static int FlashStride(int dim) => (dim + 8) * 2;

    // A property, not a field: TensorCoreNames (another file) reads it during its own static initialization.
    private static string[] FlashTensorKernels =>
        [.. new[] { 64, 128 }.SelectMany(d => new[] { $"flash_tc_fwd_d{d}", $"flash_tc_bwd_q_d{d}", $"flash_tc_bwd_kv_d{d}" })];

    private static void BuildFlashTensorCore(StringBuilder sb)
    {
        foreach (int d in new[] { 64, 128 })
        {
            FlashForward(sb, d);
            FlashBackwardQ(sb, d);
            FlashBackwardKv(sb, d);
        }
    }

    // ------------------------------------------------------------------ shared emitters

    // Declarations common to the three kernels.
    private static string FlashRegisters(int d, int accumulators) => $$"""
            .reg .pred %p<32>;
            .reg .b32 %r<100>;
            .reg .b64 %rd<48>;
            .reg .f32 %f<64>;
            .reg .f32 %acc<{{accumulators}}>;
            .reg .f32 %dv<{{accumulators}}>;
            .reg .f32 %s<32>;
            .reg .f32 %dp<32>;
            .reg .b32 %qa<{{d / 4}}>;
            .reg .b32 %ta<4>;
            .reg .b32 %tb<4>;
            .reg .b32 %pa<4>;
        """;

    // Lane-invariant setup: %r1 = tid, %r2 = lane, %r3 = warp, %r4 = g (lane / 4), %r5 = t (lane % 4); ldmatrix lane
    // offsets: %r6 = (l & 15)·stride + (l >> 4)·16 (row-major A, and [k][n] B transposed), %r7 = ((l & 7) + 8 (l >> 4))·stride
    // + ((l >> 3) & 1)·16 ([n][k] B).
    private static string FlashLanes(int stride) => $"""
            mov.u32 %r1, %tid.x;
            and.b32 %r2, %r1, 31;
            shr.u32 %r3, %r1, 5;
            shr.u32 %r4, %r2, 2;
            and.b32 %r5, %r2, 3;
            and.b32 %r8, %r2, 15;
            mul.lo.u32 %r6, %r8, {stride};
            shr.u32 %r9, %r2, 4;
            shl.b32 %r10, %r9, 4;
            add.u32 %r6, %r6, %r10;
            and.b32 %r8, %r2, 7;
            shl.b32 %r10, %r9, 3;
            add.u32 %r8, %r8, %r10;
            mul.lo.u32 %r7, %r8, {stride};
            shr.u32 %r10, %r2, 3;
            and.b32 %r10, %r10, 1;
            shl.b32 %r10, %r10, 4;
            add.u32 %r7, %r7, %r10;
        """;

    // Stages `rows` rows of a [*, d] float32 tile starting at global byte address `global` into shared memory at
    // `shared` as bfloat16 ([rows][d + 8]); rows at or beyond `valid` (a signed register) become zeros. 128 threads,
    // four floats each per step. Uses %r11-%r14, %rd1-%rd2, %f0-%f3, %p1.
    private static string FlashStage(string label, string global, string shared, int rows, int d, string valid)
    {
        int quadsPerRow = d / 4, rowStep = 128 / quadsPerRow, stride = FlashStride(d);
        var s = new StringBuilder();
        s.AppendLine($"""
                shr.u32 %r11, %r1, {(int)Math.Log2(quadsPerRow)};
                and.b32 %r12, %r1, {quadsPerRow - 1};
                shl.b32 %r12, %r12, 2;
                mul.lo.u32 %r13, %r11, {d};
                add.u32 %r13, %r13, %r12;
                mul.wide.u32 %rd1, %r13, 4;
                add.u64 %rd1, %rd1, {global};
                mul.lo.u32 %r14, %r11, {stride};
                shl.b32 %r13, %r12, 1;
                add.u32 %r14, %r14, %r13;
                add.u32 %r14, %r14, {shared};
            """);
        for (int it = 0; it < rows / rowStep; it++)
        {
            s.AppendLine($$"""
                    add.u32 %r13, %r11, {{it * rowStep}};
                    setp.lt.s32 %p1, %r13, {{valid}};
                    mov.f32 %f0, 0f00000000;
                    mov.f32 %f1, 0f00000000;
                    mov.f32 %f2, 0f00000000;
                    mov.f32 %f3, 0f00000000;
                    @%p1 ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1+{{it * rowStep * d * 4}}];
                    cvt.rn.bf16x2.f32 %r12, %f1, %f0;
                    cvt.rn.bf16x2.f32 %r13, %f3, %f2;
                    st.shared.v2.b32 [%r14+{{it * rowStep * stride}}], {%r12, %r13};
                """);
        }

        return s.ToString();
    }

    private static string Mma(string acc, int tile, string a, string b0, string b1) =>
        $"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{acc}{4 * tile}, {acc}{4 * tile + 1}, {acc}{4 * tile + 2}, {acc}{4 * tile + 3}}}, "
        + $"{{{a}0, {a}1, {a}2, {a}3}}, {{{b0}, {b1}}}, {{{acc}{4 * tile}, {acc}{4 * tile + 1}, {acc}{4 * tile + 2}, {acc}{4 * tile + 3}}};";

    private static string MmaA(string acc, int tile, string a0, string a1, string a2, string a3, string b0, string b1) =>
        $"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{acc}{4 * tile}, {acc}{4 * tile + 1}, {acc}{4 * tile + 2}, {acc}{4 * tile + 3}}}, "
        + $"{{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{{acc}{4 * tile}, {acc}{4 * tile + 1}, {acc}{4 * tile + 2}, {acc}{4 * tile + 3}}};";

    private static string LdMatrix(string dst, string address, int offset, bool transpose) =>
        $"ldmatrix.sync.aligned.m8n8.x4{(transpose ? ".trans" : "")}.shared.b16 {{{dst}0, {dst}1, {dst}2, {dst}3}}, [{address}+{offset}];";

    // acc[tiles] (a 16 × 8·tiles block, rows g and g + 8) as the A operand of k-step j: tiles 2j and 2j + 1, in bfloat16.
    private static string PackA(string acc, int j) => $"""
            cvt.rn.bf16x2.f32 %pa0, {acc}{8 * j + 1}, {acc}{8 * j};
            cvt.rn.bf16x2.f32 %pa1, {acc}{8 * j + 3}, {acc}{8 * j + 2};
            cvt.rn.bf16x2.f32 %pa2, {acc}{8 * j + 5}, {acc}{8 * j + 4};
            cvt.rn.bf16x2.f32 %pa3, {acc}{8 * j + 7}, {acc}{8 * j + 6};
        """;

    private static string Zeros(string acc, int count) => string.Concat(Enumerable.Range(0, count).Select(i => $"mov.f32 {acc}{i}, 0f00000000;\n"));

    // ------------------------------------------------------------------ forward

    // Block: 64 query rows of one head (warp w: rows 16w .. 16w + 15), walking the visible keys in tiles of 64: S = Q·Kᵀ
    // (Q fragments kept in registers), online softmax in the log2 domain, O += P·V with P straight from the accumulators.
    // Grid x = ⌈rowsPerHead / 64⌉, y = heads; 128 threads; K and V tiles 2 × 64 × (d + 8) bfloat16 of static shared memory.
    private static void FlashForward(StringBuilder sb, int d)
    {
        int stride = FlashStride(d), dTiles = d / 8, kSteps = d / 16, tileBytes = FlashTensorRows * stride;
        string name = $"flash_tc_fwd_d{d}";
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_pos, .param .u64 p_y, .param .u64 p_lse,
                .param .u32 p_rows, .param .u32 p_steps, .param .u32 p_cap, .param .f32 p_scale2
            )
            {
            {{FlashRegisters(d, 4 * dTiles)}}
                .shared .align 16 .b8 {{name}}_k[{{tileBytes}}];
                .shared .align 16 .b8 {{name}}_v[{{tileBytes}}];
            {{FlashLanes(stride)}}
                ld.param.u64 %rd10, [p_q];
                ld.param.u64 %rd11, [p_k];
                ld.param.u64 %rd12, [p_v];
                ld.param.u64 %rd13, [p_pos];
                ld.param.u64 %rd14, [p_y];
                ld.param.u64 %rd15, [p_lse];
                cvta.to.global.u64 %rd10, %rd10;
                cvta.to.global.u64 %rd11, %rd11;
                cvta.to.global.u64 %rd12, %rd12;
                cvta.to.global.u64 %rd13, %rd13;
                cvta.to.global.u64 %rd14, %rd14;
                ld.param.u32 %r20, [p_rows];
                ld.param.u32 %r21, [p_steps];
                ld.param.u32 %r22, [p_cap];
                ld.param.f32 %f40, [p_scale2];
                ld.global.f32 %f41, [%rd13];
                cvt.rzi.s32.f32 %r23, %f41;
                mov.u32 %r24, %ctaid.y;
                mov.u32 %r25, %ctaid.x;
                shl.b32 %r25, %r25, 6;
                mul.wide.u32 %rd16, %r24, %r20;
                mul.lo.u64 %rd16, %rd16, {{d * 4}};
                add.u64 %rd10, %rd10, %rd16;
                add.u64 %rd14, %rd14, %rd16;
                mul.wide.u32 %rd17, %r24, %r22;
                mul.lo.u64 %rd17, %rd17, {{d * 4}};
                add.u64 %rd11, %rd11, %rd17;
                add.u64 %rd12, %rd12, %rd17;
                mov.u32 %r26, {{name}}_k;
                mov.u32 %r27, {{name}}_v;
                mul.wide.u32 %rd18, %r25, {{d * 4}};
                add.u64 %rd18, %rd18, %rd10;
                sub.u32 %r28, %r20, %r25;
            """);
        // Q tile → shared (the K buffer) → fragments.
        s.AppendLine(FlashStage("QS", "%rd18", "%r26", FlashTensorRows, d, "%r28"));
        s.AppendLine($"""
                bar.sync 0;
                shl.b32 %r29, %r3, 4;
                mul.lo.u32 %r29, %r29, {stride};
                add.u32 %r29, %r29, %r26;
                add.u32 %r29, %r29, %r6;
            """);
        for (int ks = 0; ks < kSteps; ks++)
        {
            s.AppendLine($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, [%r29+{ks * 32}];");
        }

        // Rows of this thread: %r30 = i0, %r31 = i1; their last visible position %r32, %r33 (-1 when the row is past the end).
        // Keys to walk: %r34 = min(capacity, pos0 + (largest i % steps in the block) + 1).
        s.AppendLine($"""
                bar.sync 0;
                shl.b32 %r30, %r3, 4;
                add.u32 %r30, %r30, %r25;
                add.u32 %r30, %r30, %r4;
                add.u32 %r31, %r30, 8;
                sub.u32 %r35, %r22, 1;
                rem.u32 %r32, %r30, %r21;
                add.u32 %r32, %r32, %r23;
                min.s32 %r32, %r32, %r35;
                setp.lt.u32 %p2, %r30, %r20;
                selp.b32 %r32, %r32, -1, %p2;
                rem.u32 %r33, %r31, %r21;
                add.u32 %r33, %r33, %r23;
                min.s32 %r33, %r33, %r35;
                setp.lt.u32 %p2, %r31, %r20;
                selp.b32 %r33, %r33, -1, %p2;
                add.u32 %r36, %r25, 63;
                sub.u32 %r37, %r20, 1;
                min.u32 %r36, %r36, %r37;
                div.u32 %r38, %r25, %r21;
                div.u32 %r39, %r36, %r21;
                rem.u32 %r36, %r36, %r21;
                setp.eq.u32 %p2, %r38, %r39;
                sub.u32 %r37, %r21, 1;
                selp.b32 %r36, %r36, %r37, %p2;
                add.u32 %r34, %r36, %r23;
                add.u32 %r34, %r34, 1;
                min.u32 %r34, %r34, %r22;
                mov.f32 %f42, 0fFF800000;
                mov.f32 %f43, 0fFF800000;
                mov.f32 %f44, 0f00000000;
                mov.f32 %f45, 0f00000000;
                add.u32 %r40, %r27, %r6;
                add.u32 %r41, %r26, %r7;
                shl.b32 %r42, %r5, 1;
            {Zeros("%acc", 4 * dTiles)}
                mov.u32 %r43, 0;
            TILE:
                setp.ge.u32 %p3, %r43, %r34;
                @%p3 bra TILE_END;
                mul.wide.u32 %rd19, %r43, {d * 4};
                add.u64 %rd20, %rd19, %rd11;
                add.u64 %rd21, %rd19, %rd12;
                sub.u32 %r44, %r22, %r43;
            """);
        s.AppendLine(FlashStage("KS", "%rd20", "%r26", FlashTensorRows, d, "%r44"));
        s.AppendLine(FlashStage("VS", "%rd21", "%r27", FlashTensorRows, d, "%r44"));
        s.AppendLine("bar.sync 0;");
        s.AppendLine(Zeros("%s", 32));
        for (int ks = 0; ks < kSteps; ks++)
        {
            for (int nt2 = 0; nt2 < 4; nt2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r41", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine($"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%s{8 * nt2}, %s{8 * nt2 + 1}, %s{8 * nt2 + 2}, %s{8 * nt2 + 3}}}, "
                             + $"{{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, {{%tb0, %tb1}}, {{%s{8 * nt2}, %s{8 * nt2 + 1}, %s{8 * nt2 + 2}, %s{8 * nt2 + 3}}};");
                s.AppendLine($"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%s{8 * nt2 + 4}, %s{8 * nt2 + 5}, %s{8 * nt2 + 6}, %s{8 * nt2 + 7}}}, "
                             + $"{{%qa{4 * ks}, %qa{4 * ks + 1}, %qa{4 * ks + 2}, %qa{4 * ks + 3}}}, {{%tb2, %tb3}}, {{%s{8 * nt2 + 4}, %s{8 * nt2 + 5}, %s{8 * nt2 + 6}, %s{8 * nt2 + 7}}};");
            }
        }

        // Scale, mask (key c = tile + 8 nt + 2t + j; rows g → %r32, g + 8 → %r33), row maxima.
        s.AppendLine("""
                add.u32 %r45, %r43, %r42;
                mov.f32 %f46, 0fFF800000;
                mov.f32 %f47, 0fFF800000;
            """);
        for (int nt = 0; nt < 8; nt++)
        {
            for (int j = 0; j < 2; j++)
            {
                s.AppendLine($"""
                        add.u32 %r46, %r45, {nt * 8 + j};
                        setp.le.s32 %p4, %r46, %r32;
                        setp.le.s32 %p5, %r46, %r33;
                        mul.f32 %s{4 * nt + j}, %s{4 * nt + j}, %f40;
                        mul.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, %f40;
                        selp.f32 %s{4 * nt + j}, %s{4 * nt + j}, 0fFF800000, %p4;
                        selp.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, 0fFF800000, %p5;
                        max.f32 %f46, %f46, %s{4 * nt + j};
                        max.f32 %f47, %f47, %s{4 * nt + j + 2};
                    """);
            }
        }

        // Online softmax per row (quads share a row): new maximum, rescale factor, probabilities, row sums.
        s.AppendLine("""
                shfl.sync.bfly.b32 %f48, %f46, 1, 31, 0xffffffff;
                max.f32 %f46, %f46, %f48;
                shfl.sync.bfly.b32 %f48, %f46, 2, 31, 0xffffffff;
                max.f32 %f46, %f46, %f48;
                shfl.sync.bfly.b32 %f48, %f47, 1, 31, 0xffffffff;
                max.f32 %f47, %f47, %f48;
                shfl.sync.bfly.b32 %f48, %f47, 2, 31, 0xffffffff;
                max.f32 %f47, %f47, %f48;
                max.f32 %f46, %f46, %f42;
                max.f32 %f47, %f47, %f43;
                setp.eq.f32 %p6, %f46, 0fFF800000;
                selp.f32 %f49, 0f00000000, %f46, %p6;
                setp.eq.f32 %p7, %f47, 0fFF800000;
                selp.f32 %f50, 0f00000000, %f47, %p7;
                sub.f32 %f51, %f42, %f49;
                ex2.approx.ftz.f32 %f51, %f51;
                sub.f32 %f52, %f43, %f50;
                ex2.approx.ftz.f32 %f52, %f52;
                mov.f32 %f42, %f46;
                mov.f32 %f43, %f47;
                mov.f32 %f53, 0f00000000;
                mov.f32 %f54, 0f00000000;
            """);
        for (int nt = 0; nt < 8; nt++)
        {
            s.AppendLine($"""
                    sub.f32 %s{4 * nt}, %s{4 * nt}, %f49;
                    ex2.approx.ftz.f32 %s{4 * nt}, %s{4 * nt};
                    sub.f32 %s{4 * nt + 1}, %s{4 * nt + 1}, %f49;
                    ex2.approx.ftz.f32 %s{4 * nt + 1}, %s{4 * nt + 1};
                    sub.f32 %s{4 * nt + 2}, %s{4 * nt + 2}, %f50;
                    ex2.approx.ftz.f32 %s{4 * nt + 2}, %s{4 * nt + 2};
                    sub.f32 %s{4 * nt + 3}, %s{4 * nt + 3}, %f50;
                    ex2.approx.ftz.f32 %s{4 * nt + 3}, %s{4 * nt + 3};
                    add.f32 %f53, %f53, %s{4 * nt};
                    add.f32 %f53, %f53, %s{4 * nt + 1};
                    add.f32 %f54, %f54, %s{4 * nt + 2};
                    add.f32 %f54, %f54, %s{4 * nt + 3};
                """);
        }

        s.AppendLine("""
                shfl.sync.bfly.b32 %f48, %f53, 1, 31, 0xffffffff;
                add.f32 %f53, %f53, %f48;
                shfl.sync.bfly.b32 %f48, %f53, 2, 31, 0xffffffff;
                add.f32 %f53, %f53, %f48;
                shfl.sync.bfly.b32 %f48, %f54, 1, 31, 0xffffffff;
                add.f32 %f54, %f54, %f48;
                shfl.sync.bfly.b32 %f48, %f54, 2, 31, 0xffffffff;
                add.f32 %f54, %f54, %f48;
                fma.rn.f32 %f44, %f44, %f51, %f53;
                fma.rn.f32 %f45, %f45, %f52, %f54;
            """);
        for (int nt = 0; nt < dTiles; nt++)
        {
            s.AppendLine($"""
                    mul.f32 %acc{4 * nt}, %acc{4 * nt}, %f51;
                    mul.f32 %acc{4 * nt + 1}, %acc{4 * nt + 1}, %f51;
                    mul.f32 %acc{4 * nt + 2}, %acc{4 * nt + 2}, %f52;
                    mul.f32 %acc{4 * nt + 3}, %acc{4 * nt + 3}, %f52;
                """);
        }

        // O += P·V: k-steps over the tile's 64 keys, V [key][d] read transposed.
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine(PackA("%s", j));
            for (int dn2 = 0; dn2 < d / 16; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r40", j * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%acc", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%acc", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        s.AppendLine("""
                bar.sync 0;
                add.u32 %r43, %r43, 64;
                bra TILE;
            TILE_END:
                setp.gt.f32 %p8, %f44, 0f00000000;
                rcp.rn.f32 %f55, %f44;
                selp.f32 %f55, %f55, 0f00000000, %p8;
                setp.gt.f32 %p9, %f45, 0f00000000;
                rcp.rn.f32 %f56, %f45;
                selp.f32 %f56, %f56, 0f00000000, %p9;
                setp.lt.u32 %p10, %r30, %r20;
                setp.lt.u32 %p11, %r31, %r20;
                mul.wide.u32 %rd22, %r30, 4;
                mul.lo.u64 %rd22, %rd22, {{D}};
            """.Replace("{{D}}", d.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        s.AppendLine($"""
                add.u64 %rd22, %rd22, %rd14;
                mul.wide.u32 %rd23, %r42, 4;
                add.u64 %rd22, %rd22, %rd23;
            """);
        for (int nt = 0; nt < dTiles; nt++)
        {
            s.AppendLine($$"""
                    mul.f32 %f57, %acc{{4 * nt}}, %f55;
                    mul.f32 %f58, %acc{{4 * nt + 1}}, %f55;
                    @%p10 st.global.v2.f32 [%rd22+{{nt * 32}}], {%f57, %f58};
                    mul.f32 %f57, %acc{{4 * nt + 2}}, %f56;
                    mul.f32 %f58, %acc{{4 * nt + 3}}, %f56;
                    @%p11 st.global.v2.f32 [%rd22+{{nt * 32 + 8 * d * 4}}], {%f57, %f58};
                """);
        }

        // log-sum-exp of the scaled scores (natural log): (m + log2 l) · ln 2, by the first lane of each quad.
        s.AppendLine("""
                setp.eq.u64 %p12, %rd15, 0;
                @%p12 bra DONE;
                cvta.to.global.u64 %rd15, %rd15;
                setp.ne.u32 %p13, %r5, 0;
                @%p13 bra DONE;
                mul.wide.u32 %rd24, %r24, %r20;
                cvt.u64.u32 %rd25, %r30;
                add.u64 %rd24, %rd24, %rd25;
                shl.b64 %rd24, %rd24, 2;
                add.u64 %rd24, %rd24, %rd15;
                lg2.approx.f32 %f59, %f44;
                add.f32 %f59, %f59, %f42;
                mul.f32 %f59, %f59, 0f3F317218;
                lg2.approx.f32 %f60, %f45;
                add.f32 %f60, %f60, %f43;
                mul.f32 %f60, %f60, 0f3F317218;
                @%p10 st.global.f32 [%rd24], %f59;
                @%p11 st.global.f32 [%rd24+32], %f60;
            DONE:
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    // ------------------------------------------------------------------ backward: dQ

    // Block: 64 query rows (Q and dOutput staged once), walking the visible key tiles of 64: S = Q·Kᵀ, dP = dO·Vᵀ,
    // P = 2^(S·scale·log2 e - lse·log2 e), dS = P ∘ (dP - D), dQ += dS·K; finally dq += scale · dQ.
    // Dynamic shared memory: Q, dO, K, V tiles (4 × 64 × (d + 8) bfloat16). Grid x = ⌈rowsPerHead / 64⌉, y = heads.
    private static void FlashBackwardQ(StringBuilder sb, int d)
    {
        int stride = FlashStride(d), dTiles = d / 8, kSteps = d / 16, tile = FlashTensorRows * stride;
        string name = $"flash_tc_bwd_q_d{d}";
        var s = new StringBuilder();
        s.AppendLine($$"""
            .extern .shared .align 16 .b8 {{name}}_smem[];
            .visible .entry {{name}}(
                .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_lse, .param .u64 p_delta, .param .u64 p_dq,
                .param .u32 p_rows, .param .u32 p_steps, .param .u32 p_cap, .param .f32 p_scale, .param .f32 p_scale2
            )
            {
            {{FlashRegisters(d, 4 * dTiles)}}
            {{FlashLanes(stride)}}
                ld.param.u64 %rd10, [p_q];
                ld.param.u64 %rd11, [p_k];
                ld.param.u64 %rd12, [p_v];
                ld.param.u64 %rd13, [p_do];
                ld.param.u64 %rd14, [p_lse];
                ld.param.u64 %rd15, [p_delta];
                ld.param.u64 %rd16, [p_dq];
                cvta.to.global.u64 %rd10, %rd10;
                cvta.to.global.u64 %rd11, %rd11;
                cvta.to.global.u64 %rd12, %rd12;
                cvta.to.global.u64 %rd13, %rd13;
                cvta.to.global.u64 %rd14, %rd14;
                cvta.to.global.u64 %rd15, %rd15;
                cvta.to.global.u64 %rd16, %rd16;
                ld.param.u32 %r20, [p_rows];
                ld.param.u32 %r21, [p_steps];
                ld.param.u32 %r22, [p_cap];
                ld.param.f32 %f39, [p_scale];
                ld.param.f32 %f40, [p_scale2];
                mov.u32 %r24, %ctaid.y;
                mov.u32 %r25, %ctaid.x;
                shl.b32 %r25, %r25, 6;
                mul.wide.u32 %rd17, %r24, %r20;
                mul.lo.u64 %rd18, %rd17, {{d * 4}};
                add.u64 %rd10, %rd10, %rd18;
                add.u64 %rd13, %rd13, %rd18;
                add.u64 %rd16, %rd16, %rd18;
                shl.b64 %rd17, %rd17, 2;
                add.u64 %rd14, %rd14, %rd17;
                add.u64 %rd15, %rd15, %rd17;
                mul.wide.u32 %rd19, %r24, %r22;
                mul.lo.u64 %rd19, %rd19, {{d * 4}};
                add.u64 %rd11, %rd11, %rd19;
                add.u64 %rd12, %rd12, %rd19;
                mov.u32 %r26, {{name}}_smem;
                add.u32 %r27, %r26, {{tile}};
                add.u32 %r28, %r26, {{2 * tile}};
                add.u32 %r29, %r26, {{3 * tile}};
                mul.wide.u32 %rd20, %r25, {{d * 4}};
                add.u64 %rd21, %rd20, %rd10;
                add.u64 %rd22, %rd20, %rd13;
                sub.u32 %r30, %r20, %r25;
            """);
        s.AppendLine(FlashStage("QS", "%rd21", "%r26", FlashTensorRows, d, "%r30"));
        s.AppendLine(FlashStage("OS", "%rd22", "%r27", FlashTensorRows, d, "%r30"));
        // Rows, visible limits (causal offset 0), -lse·log2 e and D per row; keys to walk %r34.
        s.AppendLine($"""
                shl.b32 %r31, %r3, 4;
                add.u32 %r31, %r31, %r25;
                add.u32 %r31, %r31, %r4;
                add.u32 %r32, %r31, 8;
                sub.u32 %r35, %r22, 1;
                rem.u32 %r33, %r31, %r21;
                min.s32 %r33, %r33, %r35;
                setp.lt.u32 %p2, %r31, %r20;
                selp.b32 %r33, %r33, -1, %p2;
                rem.u32 %r36, %r32, %r21;
                min.s32 %r36, %r36, %r35;
                setp.lt.u32 %p3, %r32, %r20;
                selp.b32 %r36, %r36, -1, %p3;
                mul.wide.u32 %rd23, %r31, 4;
                add.u64 %rd24, %rd23, %rd14;
                add.u64 %rd25, %rd23, %rd15;
                mov.f32 %f41, 0f00000000;
                mov.f32 %f42, 0f00000000;
                mov.f32 %f43, 0f00000000;
                mov.f32 %f44, 0f00000000;
                @%p2 ld.global.f32 %f41, [%rd24];
                @%p3 ld.global.f32 %f42, [%rd24+32];
                @%p2 ld.global.f32 %f43, [%rd25];
                @%p3 ld.global.f32 %f44, [%rd25+32];
                mul.f32 %f41, %f41, 0fBFB8AA3B;
                mul.f32 %f42, %f42, 0fBFB8AA3B;
                add.u32 %r37, %r25, 63;
                sub.u32 %r38, %r20, 1;
                min.u32 %r37, %r37, %r38;
                div.u32 %r38, %r25, %r21;
                div.u32 %r39, %r37, %r21;
                rem.u32 %r37, %r37, %r21;
                setp.eq.u32 %p4, %r38, %r39;
                sub.u32 %r38, %r21, 1;
                selp.b32 %r37, %r37, %r38, %p4;
                add.u32 %r34, %r37, 1;
                min.u32 %r34, %r34, %r22;
                shl.b32 %r40, %r3, 4;
                mul.lo.u32 %r40, %r40, {stride};
                add.u32 %r40, %r40, %r6;
                add.u32 %r41, %r40, %r26;
                add.u32 %r40, %r40, %r27;
                add.u32 %r44, %r28, %r7;
                add.u32 %r45, %r29, %r7;
                add.u32 %r46, %r28, %r6;
                shl.b32 %r42, %r5, 1;
            {Zeros("%acc", 4 * dTiles)}
                mov.u32 %r43, 0;
            TILE:
                setp.ge.u32 %p5, %r43, %r34;
                @%p5 bra TILE_END;
                bar.sync 0;
                mul.wide.u32 %rd26, %r43, {d * 4};
                add.u64 %rd27, %rd26, %rd11;
                add.u64 %rd28, %rd26, %rd12;
                sub.u32 %r47, %r22, %r43;
            """);
        s.AppendLine(FlashStage("KS", "%rd27", "%r28", FlashTensorRows, d, "%r47"));
        s.AppendLine(FlashStage("VS", "%rd28", "%r29", FlashTensorRows, d, "%r47"));
        s.AppendLine("bar.sync 0;");
        s.AppendLine(Zeros("%s", 32));
        s.AppendLine(Zeros("%dp", 32));
        for (int ks = 0; ks < kSteps; ks++)
        {
            s.AppendLine(LdMatrix("%ta", "%r41", ks * 32, transpose: false));
            s.AppendLine(LdMatrix("%pa", "%r40", ks * 32, transpose: false));
            for (int nt2 = 0; nt2 < 4; nt2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r44", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine(Mma("%s", 2 * nt2, "%ta", "%tb0", "%tb1"));
                s.AppendLine(Mma("%s", 2 * nt2 + 1, "%ta", "%tb2", "%tb3"));
                s.AppendLine(LdMatrix("%tb", "%r45", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine(Mma("%dp", 2 * nt2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%dp", 2 * nt2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        // P and dS in place of S.
        s.AppendLine("add.u32 %r48, %r43, %r42;");
        for (int nt = 0; nt < 8; nt++)
        {
            for (int j = 0; j < 2; j++)
            {
                s.AppendLine($"""
                        add.u32 %r49, %r48, {nt * 8 + j};
                        setp.le.s32 %p6, %r49, %r33;
                        setp.le.s32 %p7, %r49, %r36;
                        fma.rn.f32 %s{4 * nt + j}, %s{4 * nt + j}, %f40, %f41;
                        fma.rn.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, %f40, %f42;
                        ex2.approx.ftz.f32 %s{4 * nt + j}, %s{4 * nt + j};
                        ex2.approx.ftz.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2};
                        selp.f32 %s{4 * nt + j}, %s{4 * nt + j}, 0f00000000, %p6;
                        selp.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, 0f00000000, %p7;
                        sub.f32 %dp{4 * nt + j}, %dp{4 * nt + j}, %f43;
                        sub.f32 %dp{4 * nt + j + 2}, %dp{4 * nt + j + 2}, %f44;
                        mul.f32 %s{4 * nt + j}, %s{4 * nt + j}, %dp{4 * nt + j};
                        mul.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, %dp{4 * nt + j + 2};
                    """);
            }
        }

        // dQ += dS·K (K [key][d] read transposed).
        for (int j = 0; j < 4; j++)
        {
            s.AppendLine(PackA("%s", j));
            for (int dn2 = 0; dn2 < d / 16; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r46", j * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%acc", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%acc", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        s.AppendLine($"""
                add.u32 %r43, %r43, 64;
                bra TILE;
            TILE_END:
                setp.lt.u32 %p10, %r31, %r20;
                setp.lt.u32 %p11, %r32, %r20;
                mul.wide.u32 %rd29, %r31, {d * 4};
                add.u64 %rd29, %rd29, %rd16;
                mul.wide.u32 %rd30, %r42, 4;
                add.u64 %rd29, %rd29, %rd30;
            """);
        for (int nt = 0; nt < dTiles; nt++)
        {
            s.AppendLine($$"""
                    @%p10 ld.global.v2.f32 {%f45, %f46}, [%rd29+{{nt * 32}}];
                    fma.rn.f32 %f45, %acc{{4 * nt}}, %f39, %f45;
                    fma.rn.f32 %f46, %acc{{4 * nt + 1}}, %f39, %f46;
                    @%p10 st.global.v2.f32 [%rd29+{{nt * 32}}], {%f45, %f46};
                    @%p11 ld.global.v2.f32 {%f47, %f48}, [%rd29+{{nt * 32 + 8 * d * 4}}];
                    fma.rn.f32 %f47, %acc{{4 * nt + 2}}, %f39, %f47;
                    fma.rn.f32 %f48, %acc{{4 * nt + 3}}, %f39, %f48;
                    @%p11 st.global.v2.f32 [%rd29+{{nt * 32 + 8 * d * 4}}], {%f47, %f48};
                """);
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    // ------------------------------------------------------------------ backward: dK, dV

    // Block: 64 keys (warp w: keys 16w .. 16w + 15; K and V staged once), walking the query rows that see them in tiles
    // of 32: Sᵀ = K·Qᵀ, dPᵀ = V·dOᵀ, Pᵀ = 2^(Sᵀ·scale·log2 e - lse·log2 e), dSᵀ = Pᵀ ∘ (dPᵀ - D), dV += Pᵀ·dO, dK += dSᵀ·Q;
    // finally dkeys += scale · dK, dvalues += dV. Dynamic shared memory: K, V (64 rows), Q, dO (32 rows), lse, D.
    // Grid x = ⌈capacity / 64⌉, y = heads. `skip`: rows before the block's first key see none of its keys (rowsPerHead = steps).
    private static void FlashBackwardKv(StringBuilder sb, int d)
    {
        const int R = FlashTensorBackwardRows;
        int stride = FlashStride(d), dTiles = d / 8, kSteps = d / 16, keyTile = FlashTensorRows * stride, rowTile = R * stride;
        string name = $"flash_tc_bwd_kv_d{d}";
        var s = new StringBuilder();
        s.AppendLine($$"""
            .extern .shared .align 16 .b8 {{name}}_smem[];
            .visible .entry {{name}}(
                .param .u64 p_q, .param .u64 p_k, .param .u64 p_v, .param .u64 p_do, .param .u64 p_lse, .param .u64 p_delta,
                .param .u64 p_dk, .param .u64 p_dv,
                .param .u32 p_rows, .param .u32 p_steps, .param .u32 p_cap, .param .f32 p_scale, .param .f32 p_scale2, .param .u32 p_skip
            )
            {
            {{FlashRegisters(d, 4 * dTiles)}}
            {{FlashLanes(stride)}}
                ld.param.u64 %rd10, [p_q];
                ld.param.u64 %rd11, [p_k];
                ld.param.u64 %rd12, [p_v];
                ld.param.u64 %rd13, [p_do];
                ld.param.u64 %rd14, [p_lse];
                ld.param.u64 %rd15, [p_delta];
                ld.param.u64 %rd16, [p_dk];
                ld.param.u64 %rd17, [p_dv];
                cvta.to.global.u64 %rd10, %rd10;
                cvta.to.global.u64 %rd11, %rd11;
                cvta.to.global.u64 %rd12, %rd12;
                cvta.to.global.u64 %rd13, %rd13;
                cvta.to.global.u64 %rd14, %rd14;
                cvta.to.global.u64 %rd15, %rd15;
                cvta.to.global.u64 %rd16, %rd16;
                cvta.to.global.u64 %rd17, %rd17;
                ld.param.u32 %r20, [p_rows];
                ld.param.u32 %r21, [p_steps];
                ld.param.u32 %r22, [p_cap];
                ld.param.f32 %f39, [p_scale];
                ld.param.f32 %f40, [p_scale2];
                ld.param.u32 %r23, [p_skip];
                mov.u32 %r24, %ctaid.y;
                mov.u32 %r25, %ctaid.x;
                shl.b32 %r25, %r25, 6;
                mul.wide.u32 %rd18, %r24, %r20;
                mul.lo.u64 %rd19, %rd18, {{d * 4}};
                add.u64 %rd10, %rd10, %rd19;
                add.u64 %rd13, %rd13, %rd19;
                shl.b64 %rd18, %rd18, 2;
                add.u64 %rd14, %rd14, %rd18;
                add.u64 %rd15, %rd15, %rd18;
                mul.wide.u32 %rd20, %r24, %r22;
                mul.lo.u64 %rd20, %rd20, {{d * 4}};
                add.u64 %rd11, %rd11, %rd20;
                add.u64 %rd12, %rd12, %rd20;
                add.u64 %rd16, %rd16, %rd20;
                add.u64 %rd17, %rd17, %rd20;
                mov.u32 %r26, {{name}}_smem;
                add.u32 %r27, %r26, {{keyTile}};
                add.u32 %r28, %r26, {{2 * keyTile}};
                add.u32 %r29, %r28, {{rowTile}};
                add.u32 %r30, %r29, {{rowTile}};
                mul.wide.u32 %rd21, %r25, {{d * 4}};
                add.u64 %rd22, %rd21, %rd11;
                add.u64 %rd23, %rd21, %rd12;
                sub.u32 %r31, %r22, %r25;
            """);
        s.AppendLine(FlashStage("KS", "%rd22", "%r26", FlashTensorRows, d, "%r31"));
        s.AppendLine(FlashStage("VS", "%rd23", "%r27", FlashTensorRows, d, "%r31"));
        // Keys of this thread (%r32 = key g, %r33 = key g + 8; -1 when past the end so every row masks them... handled by
        // the capacity check), fragment addresses, first row tile.
        s.AppendLine($"""
                shl.b32 %r32, %r3, 4;
                add.u32 %r32, %r32, %r25;
                add.u32 %r32, %r32, %r4;
                add.u32 %r33, %r32, 8;
                shl.b32 %r40, %r3, 4;
                mul.lo.u32 %r40, %r40, {stride};
                add.u32 %r40, %r40, %r6;
                add.u32 %r41, %r40, %r26;
                add.u32 %r40, %r40, %r27;
                add.u32 %r44, %r28, %r7;
                add.u32 %r45, %r29, %r7;
                add.u32 %r46, %r28, %r6;
                add.u32 %r47, %r29, %r6;
                shl.b32 %r42, %r5, 1;
                shr.u32 %r43, %r25, 5;
                shl.b32 %r43, %r43, 5;
                setp.eq.u32 %p2, %r23, 0;
                selp.b32 %r43, 0, %r43, %p2;
            {Zeros("%acc", 4 * dTiles)}
            {Zeros("%dv", 4 * dTiles)}
            ROWS:
                setp.ge.u32 %p3, %r43, %r20;
                @%p3 bra ROWS_END;
                bar.sync 0;
                mul.wide.u32 %rd24, %r43, {d * 4};
                add.u64 %rd25, %rd24, %rd10;
                add.u64 %rd26, %rd24, %rd13;
                sub.u32 %r48, %r20, %r43;
            """);
        s.AppendLine(FlashStage("QS", "%rd25", "%r28", R, d, "%r48"));
        s.AppendLine(FlashStage("OS", "%rd26", "%r29", R, d, "%r48"));
        // lse·log2 e (+inf past the end, so P = 0) and D of the tile's rows, by threads 0-31.
        s.AppendLine($"""
                setp.lt.u32 %p4, %r1, {R};
                @!%p4 bra STATS_DONE;
                add.u32 %r49, %r43, %r1;
                setp.lt.u32 %p5, %r49, %r20;
                mul.wide.u32 %rd27, %r49, 4;
                add.u64 %rd28, %rd27, %rd14;
                add.u64 %rd29, %rd27, %rd15;
                mov.f32 %f41, 0f7F800000;
                mov.f32 %f42, 0f00000000;
                @%p5 ld.global.f32 %f41, [%rd28];
                @%p5 ld.global.f32 %f42, [%rd29];
                @%p5 mul.f32 %f41, %f41, 0f3FB8AA3B;
                shl.b32 %r50, %r1, 2;
                add.u32 %r50, %r50, %r30;
                st.shared.f32 [%r50], %f41;
                st.shared.f32 [%r50+{R * 4}], %f42;
            STATS_DONE:
                bar.sync 0;
            {Zeros("%s", 16)}
            {Zeros("%dp", 16)}
            """);
        for (int ks = 0; ks < kSteps; ks++)
        {
            s.AppendLine(LdMatrix("%ta", "%r41", ks * 32, transpose: false));
            s.AppendLine(LdMatrix("%pa", "%r40", ks * 32, transpose: false));
            for (int nt2 = 0; nt2 < R / 16; nt2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r44", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine(Mma("%s", 2 * nt2, "%ta", "%tb0", "%tb1"));
                s.AppendLine(Mma("%s", 2 * nt2 + 1, "%ta", "%tb2", "%tb3"));
                s.AppendLine(LdMatrix("%tb", "%r45", nt2 * 16 * stride + ks * 32, transpose: false));
                s.AppendLine(Mma("%dp", 2 * nt2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%dp", 2 * nt2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        // Pᵀ (in %s) and dSᵀ (in %dp): column = row r = tile + 8 nt + 2t + j; key g (%r32) / g + 8 (%r33) visible when
        // key ≤ r % steps and key < capacity.
        s.AppendLine("""
                add.u32 %r51, %r43, %r42;
                sub.u32 %r52, %r42, 0;
            """);
        for (int nt = 0; nt < R / 8; nt++)
        {
            for (int j = 0; j < 2; j++)
            {
                s.AppendLine($"""
                        add.u32 %r53, %r51, {nt * 8 + j};
                        rem.u32 %r54, %r53, %r21;
                        setp.le.u32 %p6, %r32, %r54;
                        setp.lt.and.u32 %p6, %r32, %r22, %p6;
                        setp.le.u32 %p7, %r33, %r54;
                        setp.lt.and.u32 %p7, %r33, %r22, %p7;
                        add.u32 %r55, %r52, {nt * 8 + j};
                        shl.b32 %r55, %r55, 2;
                        add.u32 %r55, %r55, %r30;
                        ld.shared.f32 %f43, [%r55];
                        ld.shared.f32 %f44, [%r55+{R * 4}];
                        neg.f32 %f43, %f43;
                        fma.rn.f32 %s{4 * nt + j}, %s{4 * nt + j}, %f40, %f43;
                        fma.rn.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, %f40, %f43;
                        ex2.approx.ftz.f32 %s{4 * nt + j}, %s{4 * nt + j};
                        ex2.approx.ftz.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2};
                        selp.f32 %s{4 * nt + j}, %s{4 * nt + j}, 0f00000000, %p6;
                        selp.f32 %s{4 * nt + j + 2}, %s{4 * nt + j + 2}, 0f00000000, %p7;
                        sub.f32 %dp{4 * nt + j}, %dp{4 * nt + j}, %f44;
                        sub.f32 %dp{4 * nt + j + 2}, %dp{4 * nt + j + 2}, %f44;
                        mul.f32 %dp{4 * nt + j}, %dp{4 * nt + j}, %s{4 * nt + j};
                        mul.f32 %dp{4 * nt + j + 2}, %dp{4 * nt + j + 2}, %s{4 * nt + j + 2};
                    """);
            }
        }

        // dV += Pᵀ·dO and dK += dSᵀ·Q (k-steps over the 32 rows; dO and Q [row][d] read transposed).
        for (int j = 0; j < R / 16; j++)
        {
            s.AppendLine(PackA("%s", j));
            for (int dn2 = 0; dn2 < d / 16; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r47", j * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%dv", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%dv", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }

            s.AppendLine(PackA("%dp", j));
            for (int dn2 = 0; dn2 < d / 16; dn2++)
            {
                s.AppendLine(LdMatrix("%tb", "%r46", j * 16 * stride + dn2 * 32, transpose: true));
                s.AppendLine(Mma("%acc", 2 * dn2, "%pa", "%tb0", "%tb1"));
                s.AppendLine(Mma("%acc", 2 * dn2 + 1, "%pa", "%tb2", "%tb3"));
            }
        }

        s.AppendLine($"""
                add.u32 %r43, %r43, {R};
                bra ROWS;
            ROWS_END:
                setp.lt.u32 %p10, %r32, %r22;
                setp.lt.u32 %p11, %r33, %r22;
                mul.wide.u32 %rd30, %r32, {d * 4};
                mul.wide.u32 %rd31, %r42, 4;
                add.u64 %rd30, %rd30, %rd31;
                add.u64 %rd32, %rd30, %rd16;
                add.u64 %rd33, %rd30, %rd17;
            """);
        for (int nt = 0; nt < dTiles; nt++)
        {
            foreach (var (half, predicate, offset) in new[] { (0, "%p10", nt * 32), (2, "%p11", nt * 32 + 8 * d * 4) })
            {
                s.AppendLine($$"""
                        @{{predicate}} ld.global.v2.f32 {%f45, %f46}, [%rd32+{{offset}}];
                        fma.rn.f32 %f45, %acc{{4 * nt + half}}, %f39, %f45;
                        fma.rn.f32 %f46, %acc{{4 * nt + half + 1}}, %f39, %f46;
                        @{{predicate}} st.global.v2.f32 [%rd32+{{offset}}], {%f45, %f46};
                        @{{predicate}} ld.global.v2.f32 {%f47, %f48}, [%rd33+{{offset}}];
                        add.f32 %f47, %f47, %dv{{4 * nt + half}};
                        add.f32 %f48, %f48, %dv{{4 * nt + half + 1}};
                        @{{predicate}} st.global.v2.f32 [%rd33+{{offset}}], {%f47, %f48};
                    """);
            }
        }

        s.AppendLine("""
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }
}
