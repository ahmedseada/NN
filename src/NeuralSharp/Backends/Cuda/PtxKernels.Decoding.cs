using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX for fused inference kernels and incremental decoding (KV cache, device-side positions, sampling).
internal static partial class PtxKernels
{
    public static readonly string[] DecodingNames =
    [
        "scale_mask_softmax_f32", "layernorm_fused_f32", "bias_gelu_f32", "decoder_mask_f32", "kv_write_f32", "sample_rows_f32",
    ];

    private const string PosInf = "0f7F800000";

    private static void BuildDecoding(StringBuilder sb)
    {
        const string RowStart = """
            mul.lo.u32 %r5, %i, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        // softmax(scale * x + mask[row % maskrows]) per row, one thread per row.
        Elementwise(sb, "scale_mask_softmax_f32", ["x", "mask", "y"], [("u32", "cols"), ("u32", "maskrows"), ("f32", "scale"), ("u32", "hasmask")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            rem.u32 %r7, %i, %s_maskrows;
            mul.lo.u32 %r7, %r7, %s_cols;
            mul.wide.u32 %rd4, %r7, 4;
            add.u64 %rd4, %b_mask, %rd4;
            sub.u64 %rd8, %rd4, %rd2;
            setp.ne.u32 %p3, %s_hasmask, 0;
            mov.f32 %f1, {NegInf};
            """ + "\n" + RowLoop("SZ", "%rd2", """
            mul.f32 %f2, %f2, %s_scale;
            mov.f32 %f3, 0f00000000;
            add.u64 %rd9, %rd5, %rd8;
            @%p3 ld.global.f32 %f3, [%rd9];
            add.f32 %f2, %f2, %f3;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            max.f32 %f1, %f1, %f2;
            """) + "\n" + $"""
            mov.f32 %f4, {Zero};
            """ + "\n" + RowLoop("SE", "%rd3", $"""
            sub.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, {Log2E};
            ex2.approx.ftz.f32 %f2, %f2;
            st.global.f32 [%rd5], %f2;
            add.f32 %f4, %f4, %f2;
            """) + "\n" + """
            rcp.rn.f32 %f4, %f4;
            """ + "\n" + RowLoop("SN", "%rd3", """
            mul.f32 %f2, %f2, %f4;
            st.global.f32 [%rd5], %f2;
            """));

        // LayerNorm over the last dimension with gamma/beta, one thread per row.
        Elementwise(sb, "layernorm_fused_f32", ["x", "gamma", "beta", "y"], [("u32", "cols"), ("f32", "eps")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + RowLoop("LS", "%rd2", "add.f32 %f1, %f1, %f2;") + "\n" + $"""
            div.rn.f32 %f3, %f1, %f10;
            mov.f32 %f4, {Zero};
            """ + "\n" + RowLoop("LV", "%rd2", """
            sub.f32 %f5, %f2, %f3;
            fma.rn.f32 %f4, %f5, %f5, %f4;
            """) + "\n" + """
            div.rn.f32 %f4, %f4, %f10;
            add.f32 %f4, %f4, %s_eps;
            sqrt.rn.f32 %f4, %f4;
            rcp.rn.f32 %f4, %f4;
            """ + "\n" + RowLoop("LW", "%rd2", """
            mul.wide.u32 %rd7, %r6, 4;
            add.u64 %rd8, %b_gamma, %rd7;
            ld.global.f32 %f6, [%rd8];
            add.u64 %rd9, %b_beta, %rd7;
            ld.global.f32 %f7, [%rd9];
            sub.f32 %f5, %f2, %f3;
            mul.f32 %f5, %f5, %f4;
            fma.rn.f32 %f5, %f5, %f6, %f7;
            add.u64 %rd10, %rd5, %rd6;
            st.global.f32 [%rd10], %f5;
            """));

        // gelu(x + bias[i % cols]).
        Elementwise(sb, "bias_gelu_f32", ["x", "bias", "y"], [("u32", "cols")],
            $"""
            rem.u32 %r5, %i, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_bias, %rd1;
            ld.global.f32 %f1, [%a_x];
            ld.global.f32 %f2, [%rd2];
            add.f32 %f1, %f1, %f2;
            mul.f32 %f2, %f1, %f1;
            mul.f32 %f3, %f2, %f1;
            fma.rn.f32 %f4, %f3, {F(0.044715f)}, %f1;
            mul.f32 %f4, %f4, {F(0.7978845608f)};
            mul.f32 %f5, %f4, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f5, %f5;
            add.f32 %f5, %f5, {One};
            rcp.rn.f32 %f5, %f5;
            fma.rn.f32 %f6, %f5, {F(-2f)}, {One};
            add.f32 %f7, %f6, {One};
            mul.f32 %f7, %f7, %f1;
            mul.f32 %f7, %f7, {F(0.5f)};
            st.global.f32 [%a_y], %f7;
            """);

        // mask[i, j] = j <= pos + i ? 0 : -1e9 with pos read from device memory.
        Elementwise(sb, "decoder_mask_f32", ["pos", "mask"], [("u32", "cap")],
            $"""
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r5, %f1;
            div.u32 %r6, %i, %s_cap;
            rem.u32 %r7, %i, %s_cap;
            add.u32 %r8, %r5, %r6;
            setp.le.u32 %p1, %r7, %r8;
            selp.f32 %f2, {Zero}, {F(-1e9f)}, %p1;
            st.global.f32 [%a_mask], %f2;
            """);

        // cache[h, pos + t, d] = src[h, t, d]; one thread per source element.
        Elementwise(sb, "kv_write_f32", ["src", "cache", "pos"], [("u32", "stepsdim"), ("u32", "capdim"), ("u32", "dim")],
            """
            ld.global.f32 %f1, [%b_pos];
            cvt.rzi.u32.f32 %r5, %f1;
            div.u32 %r6, %i, %s_stepsdim;
            rem.u32 %r7, %i, %s_stepsdim;
            mul.lo.u32 %r8, %r6, %s_capdim;
            mad.lo.u32 %r8, %r5, %s_dim, %r8;
            add.u32 %r8, %r8, %r7;
            mul.wide.u32 %rd1, %r8, 4;
            add.u64 %rd1, %b_cache, %rd1;
            ld.global.f32 %f2, [%a_src];
            st.global.f32 [%rd1], %f2;
            """);

        SampleRows(sb);
    }

    /// <summary>Loop over the vocabulary of the current row: %r6 = index, %rd3 = address, %f2 = scaled score, %f5 = kept exp weight.</summary>
    private static string VocabularyLoop(string label, string body) => $"""
        mov.u32 %r6, 0;
        mov.u64 %rd3, %rd2;
        {label}:
        setp.ge.u32 %p1, %r6, %s_vocab;
        @%p1 bra {label}_END;
        ld.global.f32 %f2, [%rd3];
        mul.f32 %f2, %f2, %s_invt;
        setp.ge.f32 %p4, %f2, %f3;
        sub.f32 %f5, %f2, %f1;
        mul.f32 %f5, %f5, {Log2E};
        ex2.approx.ftz.f32 %f5, %f5;
        selp.f32 %f5, %f5, {Zero}, %p4;
        {body}
        {label}_NEXT:
        add.u64 %rd3, %rd3, 4;
        add.u32 %r6, %r6, 1;
        bra {label};
        {label}_END:
        """;

    /// <summary>The sampler: see Backend.SampleRows. One thread per row; mirrors CpuBackend.SampleRows step by step.</summary>
    private static void SampleRows(StringBuilder sb)
    {
        var body = new StringBuilder();
        body.AppendLine($"""
            mad.lo.u32 %r5, %i, %s_rowstride, %s_rowoffset;
            mul.wide.u32 %rd1, %r5, 4;
            add.u64 %rd2, %b_logits, %rd1;
            mov.f32 %f1, {NegInf};
            mov.u32 %r6, 0;
            mov.u64 %rd3, %rd2;
            PMAX:
            setp.ge.u32 %p1, %r6, %s_vocab;
            @%p1 bra PMAX_END;
            ld.global.f32 %f2, [%rd3];
            mul.f32 %f2, %f2, %s_invt;
            max.f32 %f1, %f1, %f2;
            add.u64 %rd3, %rd3, 4;
            add.u32 %r6, %r6, 1;
            bra PMAX;
            PMAX_END:
            mov.f32 %f3, {NegInf};
            setp.eq.u32 %p2, %s_topk, 0;
            @%p2 bra THR_DONE;
            setp.ge.u32 %p2, %s_topk, %s_vocab;
            @%p2 bra THR_DONE;
            mov.f32 %f3, {PosInf};
            mov.u32 %r7, 0;
            TK:
            setp.ge.u32 %p1, %r7, %s_topk;
            @%p1 bra THR_DONE;
            mov.f32 %f4, {NegInf};
            mov.u32 %r6, 0;
            mov.u64 %rd3, %rd2;
            TKI:
            setp.ge.u32 %p1, %r6, %s_vocab;
            @%p1 bra TKI_END;
            ld.global.f32 %f2, [%rd3];
            mul.f32 %f2, %f2, %s_invt;
            setp.lt.f32 %p3, %f2, %f3;
            setp.gt.and.f32 %p3, %f2, %f4, %p3;
            selp.f32 %f4, %f2, %f4, %p3;
            add.u64 %rd3, %rd3, 4;
            add.u32 %r6, %r6, 1;
            bra TKI;
            TKI_END:
            mov.f32 %f3, %f4;
            add.u32 %r7, %r7, 1;
            bra TK;
            THR_DONE:
            mov.f32 %f6, {Zero};
            """);
        body.AppendLine(VocabularyLoop("PSUM", "add.f32 %f6, %f6, %f5;"));
        body.AppendLine($"""
            ld.global.f32 %f7, [%b_step];
            cvt.rzi.u32.f32 %r8, %f7;
            mul.lo.u32 %r9, %r8, 0x9E3779B9;
            xor.b32 %r9, %r9, %s_seed;
            mul.lo.u32 %r10, %i, 0x85EBCA6B;
            xor.b32 %r9, %r9, %r10;
            shr.u32 %r10, %r9, 16;
            xor.b32 %r9, %r9, %r10;
            mul.lo.u32 %r9, %r9, 0x85EBCA6B;
            shr.u32 %r10, %r9, 13;
            xor.b32 %r9, %r9, %r10;
            mul.lo.u32 %r9, %r9, 0xC2B2AE35;
            shr.u32 %r10, %r9, 16;
            xor.b32 %r9, %r9, %r10;
            shr.u32 %r10, %r9, 8;
            cvt.rn.f32.u32 %f8, %r10;
            mul.f32 %f8, %f8, {F(1f / 16777216f)};
            mul.f32 %f8, %f8, %f6;
            mov.s32 %r11, -1;
            mov.f32 %f9, {Zero};
            mov.f32 %f10, {Zero};
            """);
        body.AppendLine(VocabularyLoop("PS", """
            setp.gt.f32 %p5, %f5, 0f00000000;
            @!%p5 bra PS_NEXT;
            mov.u32 %r11, %r6;
            mov.f32 %f10, %f5;
            add.f32 %f9, %f9, %f5;
            setp.gt.f32 %p6, %f9, %f8;
            @%p6 bra PS_END;
            """));
        body.AppendLine($"""
            mov.f32 %f11, {Zero};
            """);
        body.AppendLine(VocabularyLoop("PH", """
            setp.gt.f32 %p5, %f5, 0f00000000;
            @!%p5 bra PH_NEXT;
            div.rn.f32 %f12, %f5, %f6;
            lg2.approx.ftz.f32 %f13, %f12;
            mul.f32 %f13, %f13, %f12;
            sub.f32 %f11, %f11, %f13;
            """));
        body.AppendLine("""
            mad.lo.u32 %r12, %r8, %s_rows, %i;
            mul.lo.u32 %r12, %r12, 13;
            mul.wide.u32 %rd4, %r12, 4;
            add.u64 %rd5, %b_stats, %rd4;
            cvt.rn.f32.s32 %f14, %r11;
            st.global.f32 [%a_ids], %f14;
            st.global.f32 [%rd5], %f14;
            div.rn.f32 %f15, %f10, %f6;
            st.global.f32 [%rd5+4], %f15;
            st.global.f32 [%rd5+8], %f11;
            mov.s32 %r13, -1;
            mov.s32 %r14, -1;
            mov.s32 %r15, -1;
            mov.s32 %r16, -1;
            mov.s32 %r17, -1;
            """);
        for (int a = 0; a < 5; a++)
        {
            var exclude = new StringBuilder("setp.gt.f32 %p2, %f5, %f16;\n");
            for (int prev = 0; prev < a; prev++)
            {
                exclude.AppendLine($"setp.ne.s32 %p3, %r6, %r{13 + prev};");
                exclude.AppendLine("and.pred %p2, %p2, %p3;");
            }

            exclude.AppendLine("selp.f32 %f16, %f5, %f16, %p2;");
            exclude.Append("selp.b32 %r18, %r6, %r18, %p2;");
            body.AppendLine($"""
                mov.s32 %r18, -1;
                mov.f32 %f16, {Zero};
                """);
            body.AppendLine(VocabularyLoop($"TOP{a}", exclude.ToString()));
            body.AppendLine($"""
                mov.b32 %r{13 + a}, %r18;
                cvt.rn.f32.s32 %f17, %r18;
                st.global.f32 [%rd5+{12 + 8 * a}], %f17;
                div.rn.f32 %f18, %f16, %f6;
                st.global.f32 [%rd5+{16 + 8 * a}], %f18;
                """);
        }

        Elementwise(sb, "sample_rows_f32", ["logits", "ids", "stats", "step"],
            [("u32", "vocab"), ("u32", "rowstride"), ("u32", "rowoffset"), ("f32", "invt"), ("u32", "topk"), ("u32", "seed"), ("u32", "rows")],
            body.ToString());
    }
}
