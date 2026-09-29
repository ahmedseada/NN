using System.Text;

namespace NeuralSharp.Backends.Cuda;

// Products with a thin side of r ≤ 32 columns (a LoRA adapter's rank): the full-width operand is read once, from global
// memory, in tiles through shared memory, on CUDA cores in float32 (these products are bound by memory, not arithmetic),
// so every GPU runs them. The tensor-core products work on 128-column tiles, of which a rank of 16 fills an eighth.
internal static partial class PtxKernels
{
    /// <summary>The skinny products (<see cref="BuildSkinny"/>).</summary>
    public static readonly string[] SkinnyNames = ["skinny_nn_f32", "skinny_tn_f32"];

    /// <summary>Most columns of the thin side.</summary>
    public const int SkinnyMaxRank = 32;

    /// <summary>Rows (skinny_nn) or columns (skinny_tn) of the wide operand per block.</summary>
    public const int SkinnyBlock = 32;

    /// <summary>k (skinny_nn) or m (skinny_tn) per shared-memory tile; split chunks are multiples of it.</summary>
    public const int SkinnyTile = 64;

    private const int SkinnyStride = SkinnyMaxRank + 1;                  // padded rows of the thin tile: no bank conflicts

    private static void BuildSkinny(StringBuilder sb)
    {
        // y[m, r] = alpha · x[m, k] · W (+ beta · y), W = w [k, r], or w [r, k] transposed (transw). Block (x = 32 rows,
        // y = one chunk of kchunk of k); thread t: column t % 32 (idle past r) of rows 4·(t / 32) .. +3. Split chunks add
        // into y atomically (y zeroed first by the caller for beta = 0).
        sb.AppendLine(SkinnyKernel("skinny_nn_f32", transposedInput: false));

        // out = alpha · x[m, k]ᵀ · d[m, r] (+ beta · out), out [k, r], or [r, k] (transout). Block (x = 32 columns of x,
        // y = one chunk of mchunk rows); thread t: column t % 32 of r (idle past r) for x columns 4·(t / 32) .. +3.
        sb.AppendLine(SkinnyKernel("skinny_tn_f32", transposedInput: true));
    }

    private static string SkinnyKernel(string name, bool transposedInput)
    {
        const int Threads = 256, Loads = SkinnyBlock * SkinnyTile / Threads;          // 8 values of each tile per thread
        var s = new StringBuilder();
        // Registers: %r1 m, %r2 k, %r3 r, %r4 transposed thin side / output, %r5 chunk, %r6 thread, %r7 its column of r,
        // %r8 its group, %r9 first row / column of the block, %r10 chunk start, %r11 chunk end, %r12 tile start.
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_x, .param .u64 p_t, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_k, .param .u32 p_r, .param .u32 p_trans,
                .param .f32 p_alpha, .param .f32 p_beta, .param .u32 p_chunk
            )
            {
                .reg .pred %p<16>;
                .reg .pred %pc, %psplit, %pbeta;
                .reg .f32 %f<8>;
                .reg .f32 %acc<4>;
                .reg .b32 %r<48>;
                .reg .b64 %rd<16>;
                .shared .align 4 .f32 {{name}}_xs[{{SkinnyBlock * SkinnyTile}}];
                .shared .align 4 .f32 {{name}}_ts[{{SkinnyTile * SkinnyStride}}];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_t];
                ld.param.u64 %rd3, [p_y];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_k];
                ld.param.u32 %r3, [p_r];
                ld.param.u32 %r4, [p_trans];
                ld.param.u32 %r5, [p_chunk];
                mov.u32 %r6, %tid.x;
                and.b32 %r7, %r6, 31;
                shr.u32 %r8, %r6, 5;
                mov.u32 %r9, %ctaid.x;
                shl.b32 %r9, %r9, 5;
                mov.u32 %r10, %ctaid.y;
                mul.lo.u32 %r10, %r10, %r5;
                add.u32 %r11, %r10, %r5;
                min.u32 %r11, %r11, {{(transposedInput ? "%r1" : "%r2")}};
                mov.f32 %acc0, 0f00000000;
                mov.f32 %acc1, 0f00000000;
                mov.f32 %acc2, 0f00000000;
                mov.f32 %acc3, 0f00000000;
                setp.lt.u32 %pc, %r7, %r3;
                mov.u32 %r40, {{name}}_xs;
                mov.u32 %r41, {{name}}_ts;
                mov.u32 %r12, %r10;
            TILE:
                setp.ge.u32 %p1, %r12, %r11;
                @%p1 bra TILE_END;
            """);

        for (int i = 0; i < Loads; i++)
        {
            // The wide operand's tile, coalesced along its rows: skinny_nn 32 rows × 64 k (index = row · 64 + kk);
            // skinny_tn 64 rows (m) × 32 columns (index = mm · 32 + kk).
            s.AppendLine($"    add.u32 %r13, %r6, {i * Threads};");
            if (!transposedInput)
            {
                s.AppendLine("""
                        shr.u32 %r14, %r13, 6;
                        and.b32 %r15, %r13, 63;
                        add.u32 %r16, %r9, %r14;
                        add.u32 %r17, %r12, %r15;
                        setp.lt.u32 %p2, %r16, %r1;
                    """);
            }
            else
            {
                s.AppendLine("""
                        shr.u32 %r14, %r13, 5;
                        and.b32 %r15, %r13, 31;
                        add.u32 %r16, %r12, %r14;
                        add.u32 %r17, %r9, %r15;
                        setp.lt.u32 %p2, %r16, %r11;
                    """);
            }

            s.AppendLine($"""
                    setp.lt.and.u32 %p2, %r17, {(transposedInput ? "%r2" : "%r11")}, %p2;
                    mov.f32 %f1, 0f00000000;
                    mul.wide.u32 %rd4, %r16, %r2;
                    cvt.u64.u32 %rd5, %r17;
                    add.u64 %rd4, %rd4, %rd5;
                    shl.b64 %rd4, %rd4, 2;
                    add.u64 %rd4, %rd4, %rd1;
                    @%p2 ld.global.f32 %f1, [%rd4];
                    shl.b32 %r18, %r13, 2;
                    add.u32 %r18, %r18, %r40;
                    st.shared.f32 [%r18], %f1;
                """);
        }

        for (int i = 0; i < Loads; i++)
        {
            // The thin operand's tile, 64 rows (of k or of m) × r, stored at [row · 33 + j].
            s.AppendLine($"    add.u32 %r13, %r6, {i * Threads};");
            if (!transposedInput)
            {
                // w [k, r] (a contiguous run of 64 rows), or w [r, k] read transposed (transw).
                s.AppendLine("""
                        setp.ne.u32 %p3, %r4, 0;
                        @%p3 bra WT{0};
                        div.u32 %r14, %r13, %r3;
                        mul.lo.u32 %r15, %r14, %r3;
                        sub.u32 %r15, %r13, %r15;
                        add.u32 %r17, %r12, %r14;
                        mul.lo.u32 %r19, %r3, 64;
                        setp.lt.u32 %p2, %r13, %r19;
                        setp.lt.and.u32 %p2, %r17, %r11, %p2;
                        mul.wide.u32 %rd4, %r12, %r3;
                        cvt.u64.u32 %rd5, %r13;
                        add.u64 %rd4, %rd4, %rd5;
                        bra WL{0};
                    WT{0}:
                        and.b32 %r14, %r13, 63;
                        shr.u32 %r15, %r13, 6;
                        add.u32 %r17, %r12, %r14;
                        setp.lt.u32 %p2, %r15, %r3;
                        setp.lt.and.u32 %p2, %r17, %r11, %p2;
                        mul.wide.u32 %rd4, %r15, %r2;
                        cvt.u64.u32 %rd5, %r17;
                        add.u64 %rd4, %rd4, %rd5;
                    WL{0}:
                    """.Replace("{0}", i.ToString()));
            }
            else
            {
                // d [m, r]: a contiguous run of 64 rows.
                s.AppendLine("""
                        div.u32 %r14, %r13, %r3;
                        mul.lo.u32 %r15, %r14, %r3;
                        sub.u32 %r15, %r13, %r15;
                        add.u32 %r17, %r12, %r14;
                        mul.lo.u32 %r19, %r3, 64;
                        setp.lt.u32 %p2, %r13, %r19;
                        setp.lt.and.u32 %p2, %r17, %r11, %p2;
                        mul.wide.u32 %rd4, %r12, %r3;
                        cvt.u64.u32 %rd5, %r13;
                        add.u64 %rd4, %rd4, %rd5;
                    """);
            }

            s.AppendLine($"""
                    shl.b64 %rd4, %rd4, 2;
                    add.u64 %rd4, %rd4, %rd2;
                    mov.f32 %f1, 0f00000000;
                    @%p2 ld.global.f32 %f1, [%rd4];
                    setp.lt.u32 %p4, %r14, 64;
                    mad.lo.u32 %r18, %r14, {SkinnyStride}, %r15;
                    shl.b32 %r18, %r18, 2;
                    add.u32 %r18, %r18, %r41;
                    @%p4 st.shared.f32 [%r18], %f1;
                """);
        }

        // Each thread's four sums over the tile: column %r7 of the thin tile times four values of the wide tile (one
        // address per warp: a broadcast). Threads past r compute too (their sums are dropped): no divergent branch before
        // the barrier, which GPUs before Volta require.
        s.AppendLine("""
                bar.sync 0;
                shl.b32 %r30, %r7, 2;
                add.u32 %r30, %r30, %r41;
            """);
        s.AppendLine(transposedInput
            ? "    shl.b32 %r31, %r8, 4;\n    add.u32 %r31, %r31, %r40;"                 // 4 columns: 16 bytes per group
            : "    shl.b32 %r31, %r8, 10;\n    add.u32 %r31, %r31, %r40;");              // 4 rows of 64: 1024 bytes per group
        for (int kk = 0; kk < SkinnyTile; kk++)
        {
            s.AppendLine($"    ld.shared.f32 %f2, [%r30+{kk * SkinnyStride * 4}];");
            for (int i = 0; i < 4; i++)
            {
                int offset = transposedInput ? kk * SkinnyBlock * 4 + i * 4 : i * SkinnyTile * 4 + kk * 4;
                s.AppendLine($"    ld.shared.f32 %f{3 + i}, [%r31+{offset}];");
            }

            for (int i = 0; i < 4; i++)
            {
                s.AppendLine($"    fma.rn.f32 %acc{i}, %f{3 + i}, %f2, %acc{i};");
            }
        }

        s.AppendLine("""
                bar.sync 0;
                add.u32 %r12, %r12, 64;
                bra TILE;
            TILE_END:
                @!%pc bra DONE;
                mov.u32 %r20, %nctaid.y;
                setp.gt.u32 %psplit, %r20, 1;
                ld.param.f32 %f6, [p_alpha];
                ld.param.f32 %f7, [p_beta];
                setp.ne.f32 %pbeta, %f7, 0f00000000;
            """);
        for (int i = 0; i < 4; i++)
        {
            // Output element: skinny_nn y[row, j], row = first + 4·group + i; skinny_tn out[col, j] or out[j, col].
            s.AppendLine($"""
                    shl.b32 %r21, %r8, 2;
                    add.u32 %r21, %r21, {i};
                    add.u32 %r21, %r21, %r9;
                    setp.ge.u32 %p5, %r21, {(transposedInput ? "%r2" : "%r1")};
                    @%p5 bra STORED{i};
                """);
            s.AppendLine(transposedInput
                ? """
                        setp.ne.u32 %p6, %r4, 0;
                        mad.lo.u32 %r22, %r21, %r3, %r7;
                        mad.lo.u32 %r23, %r7, %r2, %r21;
                        selp.u32 %r22, %r23, %r22, %p6;
                    """
                : "    mad.lo.u32 %r22, %r21, %r3, %r7;");
            s.AppendLine($"""
                    mul.wide.u32 %rd6, %r22, 4;
                    add.u64 %rd6, %rd6, %rd3;
                    mul.f32 %f1, %acc{i}, %f6;
                    @%psplit red.global.add.f32 [%rd6], %f1;
                    @%psplit bra STORED{i};
                    @%pbeta ld.global.f32 %f2, [%rd6];
                    @%pbeta fma.rn.f32 %f1, %f2, %f7, %f1;
                    st.global.f32 [%rd6], %f1;
                STORED{i}:
                """);
        }

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        return s.ToString();
    }
}
