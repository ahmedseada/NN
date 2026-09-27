using System.Text;

namespace NeuralSharp.Backends.Cuda;

/// <summary>
/// Tensor-core matrix multiplies (sm_80 and newer: Ampere, Ada, Hopper, Blackwell), in a module of their own so older
/// GPUs still load the main one. Inputs and outputs are float32 like every other kernel: tiles are rounded to bfloat16
/// as they are copied to shared memory, multiplied with <c>mma.sync.m16n8k16</c> and summed in float32 (what PyTorch's
/// bfloat16 autocast does, without the separate cast kernels and copies).
/// </summary>
internal static partial class PtxKernels
{
    /// <summary>Output tile edge of the tensor-core GEMM (128 × 128 per block of 256 threads, 32 k per stage).</summary>
    public const int TensorTile = 128;

    /// <summary>Threads per tensor-core GEMM block.</summary>
    public const int TensorThreads = 256;

    private const int TensorK = 32;

    // Shared-memory row strides in bytes (bfloat16 rows padded by 8 elements, so ldmatrix's eight 16-byte row reads
    // fall in distinct banks): tiles 32 wide (k contiguous) and 128 wide (m or n contiguous).
    private const int NarrowStride = (TensorK + 8) * 2, WideStride = (TensorTile + 8) * 2;
    private const int StageBytes = TensorTile * NarrowStride;           // the larger of 128 × 80 and 32 × 272

    /// <summary>gemm_tc_{a}{b}_f32: a / b = n (as stored) or t (transposed), as the ta / tb flags of <c>gemm128_f32</c>.</summary>
    public static readonly string[] TensorCoreNames = ["gemm_tc_nn_f32", "gemm_tc_nt_f32", "gemm_tc_tn_f32", "gemm_tc_tt_f32", .. FlashTensorKernels];

    private static readonly Lazy<string> LazyTensorCoreSource = new(BuildTensorCore);

    /// <summary>The tensor-core module (PTX 7.0, sm_80).</summary>
    public static string TensorCoreSource => LazyTensorCoreSource.Value;

    private static readonly Lazy<IReadOnlyDictionary<string, int>> LazyTensorCoreParameterCounts = new(() =>
        System.Text.RegularExpressions.Regex.Matches(TensorCoreSource, @"\.entry\s+(\w+)\s*\(([^)]*)\)")
            .ToDictionary(m => m.Groups[1].Value, m => System.Text.RegularExpressions.Regex.Count(m.Groups[2].Value, @"\.param\b")));

    /// <summary>Parameters each tensor-core kernel declares.</summary>
    public static IReadOnlyDictionary<string, int> TensorCoreParameterCounts => LazyTensorCoreParameterCounts.Value;

    private static string BuildTensorCore()
    {
        var sb = new StringBuilder();
        sb.AppendLine(".version 7.0");
        sb.AppendLine(".target sm_80");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        foreach (bool ta in new[] { false, true })
        {
            foreach (bool tb in new[] { false, true })
            {
                TensorCoreGemm(sb, ta, tb);
            }
        }

        BuildFlashTensorCore(sb);

        return sb.ToString();
    }

    // One operand's tile as it is read from global memory: `outer` rows of `width` contiguous floats (the stored layout),
    // copied to shared memory in the same orientation. A as stored [m, k] is 128 rows of 32 k; transposed ([k, m]) 32
    // rows of 128 m. B as stored [k, n] is 32 rows of 128 n; transposed ([n, k]) 128 rows of 32 k.
    private sealed record TileLoad(string Name, bool KIsOuter, int Width, string OuterLimit, string InnerLimit, string Tile)
    {
        public int PairsPerRow => Width / 2;
        public int RowStep => 2 * TensorThreads / Width;                  // rows between one thread's successive pairs
        public int Pairs => TensorTile * TensorK / 2 / TensorThreads;     // 8 pairs per thread
        public int Stride => Width == TensorK ? NarrowStride : WideStride;
    }

    private static void TensorCoreGemm(StringBuilder sb, bool ta, bool tb)
    {
        string name = $"gemm_tc_{(ta ? 't' : 'n')}{(tb ? 't' : 'n')}_f32";
        // Registers: %rd1..3 = a, b, c; %r1..3 = m, n, k; %r9 / %r10 = the tile's first row / column.
        var a = new TileLoad("a", KIsOuter: ta, Width: ta ? TensorTile : TensorK, OuterLimit: ta ? "%r3" : "%r1", InnerLimit: ta ? "%r1" : "%r3", Tile: "%r9");
        var b = new TileLoad("b", KIsOuter: !tb, Width: tb ? TensorK : TensorTile, OuterLimit: tb ? "%r2" : "%r3", InnerLimit: tb ? "%r3" : "%r2", Tile: "%r10");
        var s = new StringBuilder();
        s.AppendLine($$"""
            .visible .entry {{name}}(
                .param .u64 p_a, .param .u64 p_b, .param .u64 p_c,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .f32 p_beta,
                .param .u64 p_sa, .param .u64 p_sb, .param .u64 p_sc
            )
            {
                .reg .pred %p<16>;
                .reg .pred %pbeta, %peven, %qnot;
                .reg .pred %q<12>;
                .reg .f32 %e<8>;
                .reg .f32 %c<64>;
                .reg .b32 %fa<16>;
                .reg .b32 %fb<8>;
                .reg .f32 %ga<16>;
                .reg .f32 %gb<16>;
                .reg .f32 %f<4>;
                .reg .f32 %beta;
                .reg .b32 %r<64>;
                .reg .b64 %rd<32>;
                .shared .align 16 .b8 {{name}}_as[{{2 * StageBytes}}];
                .shared .align 16 .b8 {{name}}_bs[{{2 * StageBytes}}];
                ld.param.u64 %rd1, [p_a];
                ld.param.u64 %rd2, [p_b];
                ld.param.u64 %rd3, [p_c];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.f32 %beta, [p_beta];
                setp.ne.f32 %pbeta, %beta, 0f00000000;
                ld.param.u64 %rd4, [p_sa];
                ld.param.u64 %rd5, [p_sb];
                ld.param.u64 %rd6, [p_sc];
                mov.u32 %r40, %ctaid.z;
                cvt.u64.u32 %rd7, %r40;
                mul.lo.u64 %rd8, %rd7, %rd4;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd1, %rd1, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd5;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd2, %rd2, %rd8;
                mul.lo.u64 %rd8, %rd7, %rd6;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd3, %rd3, %rd8;
                mov.u32 %r4, %tid.x;
                and.b32 %r5, %r4, 31;
                shr.u32 %r6, %r4, 5;
                and.b32 %r7, %r6, 1;
                shr.u32 %r8, %r6, 1;
                mov.u32 %r9, %ctaid.y;
                shl.b32 %r9, %r9, 7;
                mov.u32 %r10, %ctaid.x;
                shl.b32 %r10, %r10, 7;
                mov.u32 %r11, {{name}}_as;
                mov.u32 %r12, {{name}}_bs;
            """);

        // Per-thread load coordinates (row within the tile, first of its two columns) and shared-memory store address.
        // A: %r13 row, %r14 column, %r15 store address; B: %r16, %r17, %r18. Leading dimensions (bytes): %rd9 / %rd10.
        void Coordinates(TileLoad t, string row, string column, string store, string smem, string leading)
        {
            s.AppendLine($"""
                    shr.u32 {row}, %r4, {(int)Math.Log2(t.PairsPerRow)};
                    and.b32 {column}, %r4, {t.PairsPerRow - 1};
                    shl.b32 {column}, {column}, 1;
                    mul.lo.u32 {store}, {row}, {t.Stride};
                    shl.b32 %r19, {column}, 1;
                    add.u32 {store}, {store}, %r19;
                    add.u32 {store}, {store}, {smem};
                    mul.wide.u32 {leading}, {t.InnerLimit}, 4;
                """);
        }

        Coordinates(a, "%r13", "%r14", "%r15", "%r11", "%rd9");
        Coordinates(b, "%r16", "%r17", "%r18", "%r12", "%rd10");

        // ldmatrix lane addresses (without the stage offset), %r20 for A and %r21 for B; see the fragment layouts of
        // mma.m16n8k16 (row-major A 16 × 16, column-major B 16 × 8).
        if (!ta)
        {
            // A stored [m][k]: lanes 0-15 rows 0-15 at k 0, lanes 16-31 rows 0-15 at k 8 (non-transposed ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 15;
                    shl.b32 %r23, %r7, 6;
                    add.u32 %r22, %r22, %r23;
                    mul.lo.u32 %r20, %r22, {NarrowStride};
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 4;
                    add.u32 %r20, %r20, %r23;
                    add.u32 %r20, %r20, %r11;
                """);
        }
        else
        {
            // A stored [k][m]: lane l reads k row (l & 7) + 8 (l >> 4) at m offset 8 ((l >> 3) & 1) (transposing ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 7;
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    add.u32 %r22, %r22, %r23;
                    mul.lo.u32 %r20, %r22, {WideStride};
                    shr.u32 %r23, %r5, 3;
                    and.b32 %r23, %r23, 1;
                    shl.b32 %r23, %r23, 3;
                    shl.b32 %r24, %r7, 6;
                    add.u32 %r23, %r23, %r24;
                    shl.b32 %r23, %r23, 1;
                    add.u32 %r20, %r20, %r23;
                    add.u32 %r20, %r20, %r11;
                """);
        }

        if (tb)
        {
            // B stored [n][k]: lane l reads n row (l & 7) + 8 (l >> 4) at k offset 8 ((l >> 3) & 1) (non-transposed).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 7;
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    add.u32 %r22, %r22, %r23;
                    shl.b32 %r24, %r8, 5;
                    add.u32 %r22, %r22, %r24;
                    mul.lo.u32 %r21, %r22, {NarrowStride};
                    shr.u32 %r23, %r5, 3;
                    and.b32 %r23, %r23, 1;
                    shl.b32 %r23, %r23, 4;
                    add.u32 %r21, %r21, %r23;
                    add.u32 %r21, %r21, %r12;
                """);
        }
        else
        {
            // B stored [k][n]: lane l reads k row (l & 15) at n offset 8 (l >> 4) (transposing ldmatrix).
            s.AppendLine($"""
                    and.b32 %r22, %r5, 15;
                    mul.lo.u32 %r21, %r22, {WideStride};
                    shr.u32 %r23, %r5, 4;
                    shl.b32 %r23, %r23, 3;
                    shl.b32 %r24, %r8, 5;
                    add.u32 %r23, %r23, %r24;
                    shl.b32 %r23, %r23, 1;
                    add.u32 %r21, %r21, %r23;
                    add.u32 %r21, %r21, %r12;
                """);
        }

        for (int i = 0; i < 64; i++)
        {
            s.AppendLine($"    mov.f32 %c{i}, 0f00000000;");
        }

        // Global → registers for the k tile starting at %r30: 8 pairs of adjacent floats per operand, zero outside the matrix.
        void Load(TileLoad t, string baseAddress, string leading, string row, string column, string staging)
        {
            // outer / inner index of the thread's first pair.
            string outerBase = t.KIsOuter ? "%r30" : t.Tile, innerBase = t.KIsOuter ? t.Tile : "%r30";
            s.AppendLine($"""
                    add.u32 %r31, {outerBase}, {row};
                    add.u32 %r32, {innerBase}, {column};
                    setp.lt.u32 %p1, %r32, {t.InnerLimit};
                    add.u32 %r33, %r32, 1;
                    setp.lt.u32 %p2, %r33, {t.InnerLimit};
                    cvt.u64.u32 %rd11, %r31;
                    mul.lo.u64 %rd11, %rd11, {leading};
                    mul.wide.u32 %rd12, %r32, 4;
                    add.u64 %rd11, %rd11, %rd12;
                    add.u64 %rd11, %rd11, {baseAddress};
                    mul.lo.u64 %rd13, {leading}, {t.RowStep};
                """);
            for (int r = 0; r < t.Pairs; r++)
            {
                s.AppendLine($"""
                        setp.lt.u32 %p3, %r31, {t.OuterLimit};
                        and.pred %p4, %p3, %p1;
                        and.pred %p5, %p3, %p2;
                        mov.f32 {staging}{2 * r}, 0f00000000;
                        mov.f32 {staging}{2 * r + 1}, 0f00000000;
                        @%p4 ld.global.f32 {staging}{2 * r}, [%rd11];
                        @%p5 ld.global.f32 {staging}{2 * r + 1}, [%rd11+4];
                        add.u32 %r31, %r31, {t.RowStep};
                        add.u64 %rd11, %rd11, %rd13;
                    """);
            }
        }

        // Registers → shared memory stage %r34 (byte offset 0 or StageBytes), rounded to bfloat16 pairs.
        void Store(TileLoad t, string store, string staging)
        {
            s.AppendLine($"    add.u32 %r35, {store}, %r34;");
            for (int r = 0; r < t.Pairs; r++)
            {
                s.AppendLine($"""
                        cvt.rn.bf16x2.f32 %r36, {staging}{2 * r + 1}, {staging}{2 * r};
                        st.shared.b32 [%r35+{r * t.RowStep * t.Stride}], %r36;
                    """);
            }
        }

        s.AppendLine("""
                mov.u32 %r30, 0;
                mov.u32 %r34, 0;
            """);
        Load(a, "%rd1", "%rd9", "%r13", "%r14", "%ga");
        Load(b, "%rd2", "%rd10", "%r16", "%r17", "%gb");
        Store(a, "%r15", "%ga");
        Store(b, "%r18", "%gb");
        s.AppendLine("""
                bar.sync 0;
            KLOOP:
                setp.ge.u32 %p10, %r30, %r3;
                @%p10 bra KEND;
                add.u32 %r37, %r30, 32;
                setp.lt.u32 %p11, %r37, %r3;
                mov.u32 %r30, %r37;
                @!%p11 bra NOLOAD;
            """);
        Load(a, "%rd1", "%rd9", "%r13", "%r14", "%ga");
        Load(b, "%rd2", "%rd10", "%r16", "%r17", "%gb");
        s.AppendLine("""
            NOLOAD:
                add.u32 %r39, %r20, %r34;
                add.u32 %r41, %r21, %r34;
            """);

        // Two k16 steps over the current stage.
        for (int kk = 0; kk < 2; kk++)
        {
            for (int mt = 0; mt < 4; mt++)
            {
                int offset = ta ? kk * 16 * WideStride + mt * 16 * 2 : mt * 16 * NarrowStride + kk * 32;
                s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4{(ta ? ".trans" : "")}.shared.b16 {{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, [%r39+{offset}];");
            }

            for (int np = 0; np < 2; np++)
            {
                int offset = tb ? np * 16 * NarrowStride + kk * 32 : kk * 16 * WideStride + np * 16 * 2;
                s.AppendLine($"    ldmatrix.sync.aligned.m8n8.x4{(tb ? "" : ".trans")}.shared.b16 {{%fb{4 * np}, %fb{4 * np + 1}, %fb{4 * np + 2}, %fb{4 * np + 3}}}, [%r41+{offset}];");
            }

            for (int mt = 0; mt < 4; mt++)
            {
                for (int nt = 0; nt < 4; nt++)
                {
                    int c = (mt * 4 + nt) * 4, fb = (nt / 2) * 4 + (nt % 2) * 2;
                    s.AppendLine($"    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%c{c}, %c{c + 1}, %c{c + 2}, %c{c + 3}}}, "
                                 + $"{{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, {{%fb{fb}, %fb{fb + 1}}}, "
                                 + $"{{%c{c}, %c{c + 1}, %c{c + 2}, %c{c + 3}}};");
                }
            }
        }

        s.AppendLine($"""
                @!%p11 bra NOSTORE;
                xor.b32 %r34, %r34, {StageBytes};
            """);
        Store(a, "%r15", "%ga");
        Store(b, "%r18", "%gb");
        s.AppendLine("""
            NOSTORE:
                bar.sync 0;
                bra KLOOP;
            KEND:
            """);

        // Epilogue: c[row, col] = acc + beta · c (rows (lane >> 2) and +8 of each m16 tile, columns 2 (lane & 3) and +1
        // of each n8 tile). %r42 = first row, %r43 = first column, %rd14 = its address, %rd15 = a row in bytes.
        s.AppendLine("""
                shr.u32 %r42, %r5, 2;
                shl.b32 %r44, %r7, 6;
                add.u32 %r42, %r42, %r44;
                add.u32 %r42, %r42, %r9;
                and.b32 %r43, %r5, 3;
                shl.b32 %r43, %r43, 1;
                shl.b32 %r44, %r8, 5;
                add.u32 %r43, %r43, %r44;
                add.u32 %r43, %r43, %r10;
                mul.wide.u32 %rd15, %r2, 4;
                cvt.u64.u32 %rd14, %r42;
                mul.lo.u64 %rd14, %rd14, %rd15;
                mul.wide.u32 %rd16, %r43, 4;
                add.u64 %rd14, %rd14, %rd16;
                add.u64 %rd14, %rd14, %rd3;
            """);
        // Per row group (8 elements of one row): read the old values first, all at once (pairs as 8-byte loads when n is
        // even, so both columns share an aligned pair), then add and store. Reading them one by one between the stores
        // made the epilogue wait on memory 64 times per thread, halving the speed of beta ≠ 0 products.
        s.AppendLine("""
                and.b32 %r47, %r2, 1;
                setp.eq.u32 %peven, %r47, 0;
            """);
        for (int mt = 0; mt < 4; mt++)
        {
            for (int half = 0; half < 2; half++)
            {
                int dr = mt * 16 + half * 8;
                s.AppendLine($"""
                        add.u32 %r45, %r42, {dr};
                        setp.lt.u32 %p6, %r45, %r1;
                        mul.lo.u64 %rd17, %rd15, {dr};
                        add.u64 %rd17, %rd17, %rd14;
                    """);
                for (int nt = 0; nt < 4; nt++)
                {
                    // %q(3nt): the pair as one vector; %q(3nt+1), %q(3nt+2): single columns (odd n, or the last column).
                    s.AppendLine($"""
                            add.u32 %r46, %r43, {nt * 8};
                            setp.lt.u32 %q{3 * nt + 1}, %r46, %r2;
                            and.pred %q{3 * nt + 1}, %q{3 * nt + 1}, %p6;
                            add.u32 %r46, %r46, 1;
                            setp.lt.u32 %q{3 * nt + 2}, %r46, %r2;
                            and.pred %q{3 * nt + 2}, %q{3 * nt + 2}, %p6;
                            and.pred %q{3 * nt}, %q{3 * nt + 2}, %peven;
                            not.pred %qnot, %q{3 * nt};
                            and.pred %q{3 * nt + 1}, %q{3 * nt + 1}, %qnot;
                            and.pred %q{3 * nt + 2}, %q{3 * nt + 2}, %qnot;
                            mov.f32 %e{2 * nt}, 0f00000000;
                            mov.f32 %e{2 * nt + 1}, 0f00000000;
                        """);
                }

                s.AppendLine("@!%pbeta bra EPI_ADD_" + (mt * 2 + half) + ";");
                for (int nt = 0; nt < 4; nt++)
                {
                    s.AppendLine($$"""
                            @%q{{3 * nt}} ld.global.v2.f32 {%e{{2 * nt}}, %e{{2 * nt + 1}}}, [%rd17+{{nt * 32}}];
                            @%q{{3 * nt + 1}} ld.global.f32 %e{{2 * nt}}, [%rd17+{{nt * 32}}];
                            @%q{{3 * nt + 2}} ld.global.f32 %e{{2 * nt + 1}}, [%rd17+{{nt * 32 + 4}}];
                        """);
                }

                s.AppendLine("EPI_ADD_" + (mt * 2 + half) + ":");
                for (int nt = 0; nt < 4; nt++)
                {
                    int c = (mt * 4 + nt) * 4 + half * 2;
                    s.AppendLine($$"""
                            fma.rn.f32 %e{{2 * nt}}, %e{{2 * nt}}, %beta, %c{{c}};
                            fma.rn.f32 %e{{2 * nt + 1}}, %e{{2 * nt + 1}}, %beta, %c{{c + 1}};
                            @%q{{3 * nt}} st.global.v2.f32 [%rd17+{{nt * 32}}], {%e{{2 * nt}}, %e{{2 * nt + 1}}};
                            @%q{{3 * nt + 1}} st.global.f32 [%rd17+{{nt * 32}}], %e{{2 * nt}};
                            @%q{{3 * nt + 2}} st.global.f32 [%rd17+{{nt * 32 + 4}}], %e{{2 * nt + 1}};
                        """);
                }
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
