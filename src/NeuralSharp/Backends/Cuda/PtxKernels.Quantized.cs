using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX for int8 weight-only quantization. Weights q[k, n] are signed bytes packed four per 32-bit word along n (each
// row padded to n4 = ceil(n / 4) words), with one float scale per column: w[k, j] = q[k, j] · scale[j].
internal static partial class PtxKernels
{
    public static readonly string[] QuantizedNames = ["int8_matmul_f32", "int8_dequant_f32", "kv_write_int8", "attn_scores_int8", "attn_context_int8", "int8_gemv_f32", "int8_gemv_finish_f32"];

    /// <summary>Threads of an <c>int8_gemv_f32</c> block: 32 packed words (128 columns) × 16 slices of k.</summary>
    public const int Int8GemvThreads = 512;

    private static void BuildQuantized(StringBuilder sb)
    {
        Int8Gemv(sb);

        // y[i] = scale[i % cols] · Σ_s part[s, i] over the k splits of int8_gemv_f32, in split order.
        Elementwise(sb, "int8_gemv_finish_f32", ["part", "s", "y"], [("u32", "cols"), ("u32", "size"), ("u32", "splits")], $"""
            mov.f32 %f1, {Zero};
            mov.u64 %rd2, %a_part;
            mul.wide.u32 %rd3, %s_size, 4;
            mov.u32 %r5, 0;
            FS:
            setp.ge.u32 %p1, %r5, %s_splits;
            @%p1 bra FS_END;
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            add.u64 %rd2, %rd2, %rd3;
            add.u32 %r5, %r5, 1;
            bra FS;
            FS_END:
            rem.u32 %r6, %i, %s_cols;
            mul.wide.u32 %rd4, %r6, 4;
            add.u64 %rd4, %rd4, %b_s;
            ld.global.f32 %f3, [%rd4];
            mul.f32 %f1, %f1, %f3;
            st.global.f32 [%a_y], %f1;
            """);

        // y[r, 4c..4c+3] = scale · Σ_k x[r, k] · q[k, 4c..4c+3]: one thread per (row, word). Adjacent threads read adjacent
        // words of a weight row, so a warp reads 128 contiguous bytes per step (the weights are read once per row).
        Elementwise(sb, "int8_matmul_f32", ["x", "q", "s", "y"], [("u32", "k"), ("u32", "n4"), ("u32", "cols")], $"""
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            mul.lo.u32 %r7, %r5, %s_k;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd2, %b_x, %rd1;
            mul.wide.u32 %rd3, %r6, 4;
            add.u64 %rd4, %b_q, %rd3;
            mul.wide.u32 %rd5, %s_n4, 4;
            mov.f32 %f1, {Zero};
            mov.f32 %f2, {Zero};
            mov.f32 %f3, {Zero};
            mov.f32 %f4, {Zero};
            mov.u32 %r8, 0;
            QM_LOOP:
            setp.ge.u32 %p1, %r8, %s_k;
            @%p1 bra QM_END;
            ld.global.f32 %f5, [%rd2];
            ld.global.u32 %r9, [%rd4];
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f1, %f5, %f6, %f1;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f2, %f5, %f6, %f2;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f3, %f5, %f6, %f3;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f4, %f5, %f6, %f4;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, %rd5;
            add.u32 %r8, %r8, 1;
            bra QM_LOOP;
            QM_END:
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_y, %rd8;
            """ + "\n" + StoreScaledColumns("QM", "%f1", "%f2", "%f3", "%f4"));

        // w[kk, 4c..4c+3] = q[kk, 4c..4c+3] · scale: one thread per packed word.
        Elementwise(sb, "int8_dequant_f32", ["q", "s", "w"], [("u32", "n4"), ("u32", "cols")], """
            ld.global.u32 %r9, [%a_q];
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f1, %r10;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f2, %r10;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f3, %r10;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f4, %r10;
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_w, %rd8;
            """ + "\n" + StoreScaledColumns("QD", "%f1", "%f2", "%f3", "%f4"));

        // Int8 KV cache. Each cached row (one head, one position) holds dim values as bytes packed four per word
        // (words = ceil(dim / 4)) and one scale (max |x| / 127). The position comes from the device.

        // Quantizes source rows [heads·steps, dim] into slots [head, position + step]: one thread per row.
        Elementwise(sb, "kv_write_int8", ["src", "cache", "scales", "position"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            div.u32 %r5, %i, %s_steps;
            rem.u32 %r6, %i, %s_steps;
            ld.global.f32 %f1, [%b_position];
            cvt.rzi.u32.f32 %r7, %f1;
            add.u32 %r7, %r7, %r6;
            mul.lo.u32 %r8, %i, %s_dim;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_src, %rd1;
            mov.f32 %f2, {{Zero}};
            mov.u32 %r9, 0;
            KW_MAX:
            setp.ge.u32 %p1, %r9, %s_dim;
            @%p1 bra KW_MAXED;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %rd2, %rd3;
            ld.global.f32 %f3, [%rd4];
            abs.f32 %f3, %f3;
            max.f32 %f2, %f2, %f3;
            add.u32 %r9, %r9, 1;
            bra KW_MAX;
            KW_MAXED:
            div.rn.f32 %f4, %f2, 0f42FE0000;
            setp.gt.f32 %p2, %f2, {{Zero}};
            mov.f32 %f5, {{Zero}};
            @%p2 rcp.rn.f32 %f5, %f4;
            mul.lo.u32 %r10, %r5, %s_capacity;
            add.u32 %r10, %r10, %r7;
            mul.wide.u32 %rd5, %r10, 4;
            add.u64 %rd6, %b_scales, %rd5;
            st.global.f32 [%rd6], %f4;
            mul.lo.u32 %r11, %r10, %s_words;
            mul.wide.u32 %rd7, %r11, 4;
            add.u64 %rd8, %b_cache, %rd7;
            mov.u32 %r12, 0;
            KW_WORD:
            setp.ge.u32 %p3, %r12, %s_words;
            @%p3 bra KW_DONE;
            mov.u32 %r13, 0;
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => $"""
            shl.b32 %r14, %r12, 2;
            add.u32 %r14, %r14, {t};
            setp.ge.u32 %p4, %r14, %s_dim;
            mov.f32 %f6, {Zero};
            mul.wide.u32 %rd9, %r14, 4;
            add.u64 %rd10, %rd2, %rd9;
            @!%p4 ld.global.f32 %f6, [%rd10];
            mul.f32 %f7, %f6, %f5;
            cvt.rni.s32.f32 %r15, %f7;
            min.s32 %r15, %r15, 127;
            max.s32 %r15, %r15, -127;
            and.b32 %r15, %r15, 255;
            shl.b32 %r15, %r15, {8 * t};
            or.b32 %r13, %r13, %r15;

            """)) + """
            mul.wide.u32 %rd11, %r12, 4;
            add.u64 %rd12, %rd8, %rd11;
            st.global.u32 [%rd12], %r13;
            add.u32 %r12, %r12, 1;
            bra KW_WORD;
            KW_DONE:
            """);

        // scores[r, t, c] = scale[r, c] · Σ_d q[r, t, d] · k[r, c, d]: one thread per (r, t, c).
        Elementwise(sb, "attn_scores_int8", ["q", "cache", "scales", "y"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            rem.u32 %r5, %i, %s_capacity;
            div.u32 %r6, %i, %s_capacity;
            div.u32 %r7, %r6, %s_steps;
            mul.lo.u32 %r8, %r6, %s_dim;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_q, %rd1;
            mul.lo.u32 %r9, %r7, %s_capacity;
            add.u32 %r9, %r9, %r5;
            mul.lo.u32 %r10, %r9, %s_words;
            mul.wide.u32 %rd3, %r10, 4;
            add.u64 %rd4, %b_cache, %rd3;
            mov.f32 %f1, {{Zero}};
            mov.u32 %r11, 0;
            AS_LOOP:
            setp.ge.u32 %p1, %r11, %s_words;
            @%p1 bra AS_END;
            ld.global.u32 %r12, [%rd4];
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => $"""
            shl.b32 %r13, %r11, 2;
            add.u32 %r13, %r13, {t};
            setp.ge.u32 %p2, %r13, %s_dim;
            @%p2 bra AS_NEXT;
            mul.wide.u32 %rd5, %r13, 4;
            add.u64 %rd6, %rd2, %rd5;
            ld.global.f32 %f2, [%rd6];
            bfe.s32 %r14, %r12, {8 * t}, 8;
            cvt.rn.f32.s32 %f3, %r14;
            fma.rn.f32 %f1, %f2, %f3, %f1;

            """)) + """
            AS_NEXT:
            add.u64 %rd4, %rd4, 4;
            add.u32 %r11, %r11, 1;
            bra AS_LOOP;
            AS_END:
            mul.wide.u32 %rd7, %r9, 4;
            add.u64 %rd8, %b_scales, %rd7;
            ld.global.f32 %f4, [%rd8];
            mul.f32 %f1, %f1, %f4;
            st.global.f32 [%a_y], %f1;
            """);

        // out[r, t, 4w..4w+3] = Σ_c weights[r, t, c] · scale[r, c] · v[r, c, 4w..4w+3]: one thread per (r, t, word).
        Elementwise(sb, "attn_context_int8", ["w", "cache", "scales", "y"], [("u32", "steps"), ("u32", "capacity"), ("u32", "dim"), ("u32", "words")], $$"""
            rem.u32 %r5, %i, %s_words;
            div.u32 %r6, %i, %s_words;
            div.u32 %r7, %r6, %s_steps;
            mul.lo.u32 %r8, %r6, %s_capacity;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd2, %b_w, %rd1;
            mul.lo.u32 %r9, %r7, %s_capacity;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %b_scales, %rd3;
            mul.lo.u32 %r10, %r9, %s_words;
            add.u32 %r10, %r10, %r5;
            mul.wide.u32 %rd5, %r10, 4;
            add.u64 %rd6, %b_cache, %rd5;
            mul.wide.u32 %rd7, %s_words, 4;
            mov.f32 %f1, {{Zero}};
            mov.f32 %f2, {{Zero}};
            mov.f32 %f3, {{Zero}};
            mov.f32 %f4, {{Zero}};
            mov.u32 %r11, 0;
            AC_LOOP:
            setp.ge.u32 %p1, %r11, %s_capacity;
            @%p1 bra AC_END;
            ld.global.f32 %f5, [%rd2];
            ld.global.f32 %f6, [%rd4];
            mul.f32 %f5, %f5, %f6;
            ld.global.u32 %r12, [%rd6];
            bfe.s32 %r13, %r12, 0, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f1, %f5, %f7, %f1;
            bfe.s32 %r13, %r12, 8, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f2, %f5, %f7, %f2;
            bfe.s32 %r13, %r12, 16, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f3, %f5, %f7, %f3;
            bfe.s32 %r13, %r12, 24, 8;
            cvt.rn.f32.s32 %f7, %r13;
            fma.rn.f32 %f4, %f5, %f7, %f4;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, 4;
            add.u64 %rd6, %rd6, %rd7;
            add.u32 %r11, %r11, 1;
            bra AC_LOOP;
            AC_END:
            shl.b32 %r14, %r5, 2;
            mul.lo.u32 %r15, %r6, %s_dim;
            add.u32 %r15, %r15, %r14;
            mul.wide.u32 %rd8, %r15, 4;
            add.u64 %rd9, %b_y, %rd8;
            """ + "\n" + string.Concat(Enumerable.Range(0, 4).Select(t => $"""
            add.u32 %r13, %r14, {t};
            setp.ge.u32 %p2, %r13, %s_dim;
            @%p2 bra AC_STORED;
            st.global.f32 [%rd9+{4 * t}], %f{t + 1};

            """)) + "AC_STORED:");
    }

    // Stores value[t] · scale[column + t] for the (up to) four columns starting at %r11 that are below n.
    // Expects %rd7 = &scale[column] and %rd9 = &out[row, column].
    private static string StoreScaledColumns(string prefix, params string[] values)
    {
        var sb = new StringBuilder();
        for (int t = 0; t < values.Length; t++)
        {
            sb.AppendLine($"add.u32 %r13, %r11, {t};");
            sb.AppendLine("setp.ge.u32 %p2, %r13, %s_cols;");
            sb.AppendLine($"@%p2 bra {prefix}_STORED;");
            sb.AppendLine($"ld.global.f32 %f7, [%rd7+{4 * t}];");
            sb.AppendLine($"mul.f32 %f8, {values[t]}, %f7;");
            sb.AppendLine($"st.global.f32 [%rd9+{4 * t}], %f8;");
        }

        sb.Append($"{prefix}_STORED:");
        return sb.ToString();
    }

    // y[r, j] = scale[j] · Σ_k x[r, k] · q[k, j] for m ≤ 8 rows, reading each packed weight word once. Block: 32 lanes
    // (a word = 4 columns each, so a warp reads 128 contiguous bytes) × 16 slices of k, four k per slice in flight;
    // grid: x = ⌈words / 32⌉, y = k splits of `chunk` rows each (enough blocks to fill the GPU when the matrix is
    // narrow). Slices are added in shared memory; with one split the result is scaled and stored, otherwise the
    // split's partial sums go to part[split, r, j] and int8_gemv_finish_f32 adds them in order.
    private static void Int8Gemv(StringBuilder sb)
    {
        var s = new StringBuilder();
        s.AppendLine("""
            .visible .entry int8_gemv_f32(
                .param .u64 p_x, .param .u64 p_q, .param .u64 p_s, .param .u64 p_y, .param .u64 p_part,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_n4, .param .u32 p_chunk, .param .u32 p_splits
            )
            {
                .reg .pred %p<24>;
                .reg .f32 %f<48>;
                .reg .b32 %r<40>;
                .reg .b64 %rd<24>;
                .shared .align 4 .f32 i8_part[2048];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_q];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                ld.param.u64 %rd5, [p_part];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.u32 %r4, [p_n4];
                ld.param.u32 %r30, [p_chunk];
                ld.param.u32 %r31, [p_splits];
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 31;
                shr.u32 %r7, %r5, 5;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 5;
                add.u32 %r8, %r8, %r6;
                setp.lt.u32 %p20, %r8, %r4;
                mov.u32 %r9, %ctaid.y;
                mul.lo.u32 %r10, %r9, %r30;
                add.u32 %r11, %r10, %r30;
                min.u32 %r11, %r11, %r3;
                mul.wide.u32 %rd6, %r4, 64;
                mul.wide.u32 %rd7, %r3, 4;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            s.AppendLine($"    setp.lt.u32 %p{r}, {r}, %r1;");
            for (int c = 0; c < 4; c++)
            {
                s.AppendLine($"    mov.f32 %f{r * 4 + c}, 0f00000000;");
            }
        }

        // One k step: word %r{word} holds 4 columns; x[r, kk] is at %rd{xaddr} + {offset} + r·k·4.
        string Step(string wordReg, string xOffset)
        {
            var t = new StringBuilder();
            for (int c = 0; c < 4; c++)
            {
                t.AppendLine($"bfe.s32 %r20, {wordReg}, {8 * c}, 8;");
                t.AppendLine($"cvt.rn.f32.s32 %f{32 + c}, %r20;");
            }

            t.AppendLine("mov.u64 %rd12, %rd11;");
            for (int r = 0; r < GemvRows; r++)
            {
                t.AppendLine($"@%p{r} ld.global.f32 %f36, [%rd12+{xOffset}];");
                for (int c = 0; c < 4; c++)
                {
                    t.AppendLine($"@%p{r} fma.rn.f32 %f{r * 4 + c}, %f36, %f{32 + c}, %f{r * 4 + c};");
                }

                t.AppendLine("add.u64 %rd12, %rd12, %rd7;");
            }

            return t.ToString();
        }

        s.AppendLine("""
                add.u32 %r12, %r10, %r7;
            K4:
                add.u32 %r13, %r12, 48;
                setp.ge.u32 %p10, %r13, %r11;
                @%p10 bra K1;
                mul.wide.u32 %rd9, %r12, %r4;
                cvt.u64.u32 %rd10, %r8;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                mov.u32 %r21, 0;
                mov.u32 %r22, 0;
                mov.u32 %r23, 0;
                mov.u32 %r24, 0;
                @%p20 ld.global.u32 %r21, [%rd9];
                add.u64 %rd9, %rd9, %rd6;
                @%p20 ld.global.u32 %r22, [%rd9];
                add.u64 %rd9, %rd9, %rd6;
                @%p20 ld.global.u32 %r23, [%rd9];
                add.u64 %rd9, %rd9, %rd6;
                @%p20 ld.global.u32 %r24, [%rd9];
                mul.wide.u32 %rd11, %r12, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        s.AppendLine(Step("%r21", "0"));
        s.AppendLine(Step("%r22", "64"));
        s.AppendLine(Step("%r23", "128"));
        s.AppendLine(Step("%r24", "192"));
        s.AppendLine("""
                add.u32 %r12, %r12, 64;
                bra K4;
            K1:
                setp.ge.u32 %p10, %r12, %r11;
                @%p10 bra KEND;
                mul.wide.u32 %rd9, %r12, %r4;
                cvt.u64.u32 %rd10, %r8;
                add.u64 %rd9, %rd9, %rd10;
                shl.b64 %rd9, %rd9, 2;
                add.u64 %rd9, %rd9, %rd2;
                mov.u32 %r21, 0;
                @%p20 ld.global.u32 %r21, [%rd9];
                mul.wide.u32 %rd11, %r12, 4;
                add.u64 %rd11, %rd11, %rd1;
            """);
        s.AppendLine(Step("%r21", "0"));
        s.AppendLine("""
                add.u32 %r12, %r12, 16;
                bra K1;
            KEND:
                mov.u32 %r14, i8_part;
                shl.b32 %r15, %r7, 9;
                shl.b32 %r16, %r6, 4;
                add.u32 %r15, %r15, %r16;
                add.u32 %r15, %r15, %r14;
                shl.b32 %r17, %r5, 2;
                add.u32 %r17, %r17, %r14;
                mov.u32 %r18, %ctaid.x;
                shl.b32 %r18, %r18, 7;
                add.u32 %r18, %r18, %r5;
                setp.lt.u32 %p11, %r5, 128;
                setp.lt.u32 %p12, %r18, %r2;
                and.pred %p11, %p11, %p12;
                setp.eq.u32 %p13, %r31, 1;
                mul.wide.u32 %rd13, %r18, 4;
                add.u64 %rd14, %rd13, %rd3;
            """);
        for (int r = 0; r < GemvRows; r++)
        {
            // part[slice][lane·4 + c] = acc[r][c]; then threads 0..127 add the 16 slices of their column.
            s.AppendLine($"    @!%p{r} bra DONE;");
            for (int c = 0; c < 4; c++)
            {
                s.AppendLine($"    st.shared.f32 [%r15+{4 * c}], %f{r * 4 + c};");
            }

            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    @!%p11 bra ROW{r}_DONE;");
            s.AppendLine("    mov.f32 %f40, 0f00000000;");
            for (int slice = 0; slice < 16; slice++)
            {
                s.AppendLine($"    ld.shared.f32 %f41, [%r17+{slice * 512}];");
                s.AppendLine("    add.f32 %f40, %f40, %f41;");
            }

            s.AppendLine($"""
                    mad.lo.u32 %r25, %r9, %r1, {r};
                    mul.wide.u32 %rd15, %r25, %r2;
                    @%p13 bra ROW{r}_FINAL;
                    shl.b64 %rd16, %rd15, 2;
                    add.u64 %rd16, %rd16, %rd13;
                    add.u64 %rd16, %rd16, %rd5;
                    st.global.f32 [%rd16], %f40;
                    bra ROW{r}_DONE;
                ROW{r}_FINAL:
                    ld.global.f32 %f42, [%rd14];
                    mul.f32 %f40, %f40, %f42;
                    mul.wide.u32 %rd16, %r2, {4 * r};
                    add.u64 %rd16, %rd16, %rd13;
                    add.u64 %rd16, %rd16, %rd4;
                    st.global.f32 [%rd16], %f40;
                ROW{r}_DONE:
                    bar.sync 0;
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
