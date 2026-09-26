using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX for decoder-only language model layers: RMS normalization and rotary position embeddings.
internal static partial class PtxKernels
{
    public static readonly string[] DecoderNames = ["rms_norm_f32", "rms_norm_backward_f32", "rope_f32", "rms_norm_affine_f32", "gated_act_f32", "gated_act_bwd_f32"];

    private static void BuildDecoder(StringBuilder sb)
    {
        const string RowStart = """
            mul.lo.u32 %r5, %row, %s_cols;
            mul.wide.u32 %rd1, %r5, 4;
            """;

        // y = x · inv, inv = 1 / sqrt(mean(x²) + eps) per row (stored for the backward pass): one block per row.
        RowBlock(sb, "rms_norm_f32", ["x", "y", "inv"], [("u32", "cols"), ("f32", "eps")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            setp.eq.u32 %p5, %tx, 0;
            @%p5 st.global.f32 [%a_inv], %f1;
            """ + "\n" + StridedLoop("RW", "%rd2", "%s_cols", """
            mul.f32 %f2, %f2, %f1;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """));

        // dx += inv · (dy - y · mean(dy · y)) per row (y is the normalized output): one block per row.
        RowBlock(sb, "rms_norm_backward_f32", ["dy", "y", "inv", "dx"], [("u32", "cols")],
            RowStart + $"""
            add.u64 %rd2, %b_dy, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            add.u64 %rd4, %b_dx, %rd1;
            sub.u64 %rd8, %rd4, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RD", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            fma.rn.f32 %f1, %f2, %f3, %f1;
            """) + "\n" + BlockReduce("RDOT", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            ld.global.f32 %f6, [%a_inv];
            """ + "\n" + StridedLoop("RB", "%rd2", "%s_cols", """
            add.u64 %rd7, %rd5, %rd6;
            ld.global.f32 %f3, [%rd7];
            mul.f32 %f4, %f3, %f1;
            sub.f32 %f4, %f2, %f4;
            mul.f32 %f4, %f4, %f6;
            add.u64 %rd9, %rd5, %rd8;
            ld.global.f32 %f5, [%rd9];
            add.f32 %f5, %f5, %f4;
            st.global.f32 [%rd9], %f5;
            """));

        // y = x · inv · (gain + offset): normalization and gain in one pass (inference), one block per row.
        RowBlock(sb, "rms_norm_affine_f32", ["x", "gain", "y"], [("u32", "cols"), ("f32", "eps"), ("f32", "offset")],
            RowStart + $"""
            add.u64 %rd2, %b_x, %rd1;
            add.u64 %rd3, %b_y, %rd1;
            sub.u64 %rd6, %rd3, %rd2;
            cvt.rn.f32.u32 %f10, %s_cols;
            mov.f32 %f1, {Zero};
            """ + "\n" + StridedLoop("RS", "%rd2", "%s_cols", "fma.rn.f32 %f1, %f2, %f2, %f1;") + "\n"
            + BlockReduce("RSUM", "%f1", "add", Zero) + "\n" + """
            div.rn.f32 %f1, %f1, %f10;
            add.f32 %f1, %f1, %s_eps;
            sqrt.rn.f32 %f1, %f1;
            rcp.rn.f32 %f1, %f1;
            """ + "\n" + StridedLoop("RW", "%rd2", "%s_cols", """
            mul.wide.u32 %rd8, %r6, 4;
            add.u64 %rd8, %rd8, %b_gain;
            ld.global.f32 %f3, [%rd8];
            add.f32 %f3, %f3, %s_offset;
            mul.f32 %f2, %f2, %f1;
            mul.f32 %f2, %f2, %f3;
            add.u64 %rd7, %rd5, %rd6;
            st.global.f32 [%rd7], %f2;
            """));

        // act(gate) and act'(gate) into %f5 and %f6 (kind 0 SiLU, 1 GELU tanh, 2 ReLU) from %f1.
        string activation = $"""
            setp.eq.u32 %p5, %s_kind, 1;
            @%p5 bra ACT_GELU;
            setp.eq.u32 %p5, %s_kind, 2;
            @%p5 bra ACT_RELU;
            mul.f32 %f3, %f1, {F(-1.4426950408889634f)};
            ex2.approx.ftz.f32 %f3, %f3;
            add.f32 %f3, %f3, {One};
            rcp.rn.f32 %f3, %f3;
            mul.f32 %f5, %f1, %f3;
            sub.f32 %f4, {One}, %f3;
            fma.rn.f32 %f4, %f1, %f4, {One};
            mul.f32 %f6, %f3, %f4;
            bra ACT_DONE;
            ACT_GELU:
            mul.f32 %f7, %f1, %f1;
            mul.f32 %f8, %f7, %f1;
            fma.rn.f32 %f8, %f8, {F(0.044715f)}, %f1;
            mul.f32 %f8, %f8, {F(0.7978845608f)};
            mul.f32 %f9, %f8, {F(2.8853900817779268f)};
            ex2.approx.ftz.f32 %f9, %f9;
            add.f32 %f9, %f9, {One};
            rcp.rn.f32 %f9, %f9;
            fma.rn.f32 %f9, %f9, {F(-2f)}, {One};
            add.f32 %f10, %f9, {One};
            mul.f32 %f5, %f10, %f1;
            mul.f32 %f5, %f5, {F(0.5f)};
            mul.f32 %f11, %f9, %f9;
            sub.f32 %f11, {One}, %f11;
            mul.f32 %f11, %f11, %f1;
            mul.f32 %f11, %f11, {F(0.5f * 0.7978845608f)};
            fma.rn.f32 %f12, %f7, {F(3f * 0.044715f)}, {One};
            mul.f32 %f11, %f11, %f12;
            fma.rn.f32 %f6, %f10, {F(0.5f)}, %f11;
            bra ACT_DONE;
            ACT_RELU:
            max.f32 %f5, %f1, {Zero};
            setp.gt.f32 %p6, %f1, {Zero};
            selp.f32 %f6, {One}, {Zero}, %p6;
            ACT_DONE:
            """;

        // y = act(gate) · up.
        Elementwise(sb, "gated_act_f32", ["gate", "up", "y"], [("u32", "kind")],
            """
            ld.global.f32 %f1, [%a_gate];
            ld.global.f32 %f2, [%a_up];
            """ + "\n" + activation + "\n" + """
            mul.f32 %f5, %f5, %f2;
            st.global.f32 [%a_y], %f5;
            """);

        // dgate += dy · up · act'(gate) (flags & 1), dup += dy · act(gate) (flags & 2).
        Elementwise(sb, "gated_act_bwd_f32", ["gate", "up", "dy", "dgate", "dup"], [("u32", "kind"), ("u32", "flags")],
            """
            ld.global.f32 %f1, [%a_gate];
            ld.global.f32 %f2, [%a_up];
            ld.global.f32 %f13, [%a_dy];
            """ + "\n" + activation + "\n" + """
            and.b32 %r5, %s_flags, 1;
            setp.eq.u32 %p7, %r5, 0;
            @%p7 bra NO_DGATE;
            mul.f32 %f14, %f13, %f2;
            ld.global.f32 %f15, [%a_dgate];
            fma.rn.f32 %f15, %f14, %f6, %f15;
            st.global.f32 [%a_dgate], %f15;
            NO_DGATE:
            and.b32 %r5, %s_flags, 2;
            setp.eq.u32 %p7, %r5, 0;
            @%p7 bra NO_DUP;
            ld.global.f32 %f16, [%a_dup];
            fma.rn.f32 %f16, %f13, %f5, %f16;
            st.global.f32 [%a_dup], %f16;
            NO_DUP:
            """);

        // Rotary embedding of x [rows = batch·steps·heads, dim]: one thread per (row, pair). The pair of pair index p is
        // (2p, 2p+1) when interleaved, else (p, p + half). y must already hold x (dimensions beyond 2·half pass through).
        // sign = -1 applies the inverse rotation (the backward pass).
        Elementwise(sb, "rope_f32", ["x", "y", "cos", "sin", "positions"],
            [("u32", "heads"), ("u32", "steps"), ("u32", "dim"), ("u32", "half"), ("u32", "interleaved"), ("f32", "sign")], """
            div.u32 %r5, %i, %s_half;
            rem.u32 %r6, %i, %s_half;
            div.u32 %r7, %r5, %s_heads;
            rem.u32 %r7, %r7, %s_steps;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd2, %b_positions, %rd1;
            ld.global.f32 %f1, [%rd2];
            cvt.rzi.u32.f32 %r8, %f1;
            mad.lo.u32 %r9, %r8, %s_half, %r6;
            mul.wide.u32 %rd3, %r9, 4;
            add.u64 %rd4, %b_cos, %rd3;
            ld.global.f32 %f2, [%rd4];
            add.u64 %rd4, %b_sin, %rd3;
            ld.global.f32 %f3, [%rd4];
            mul.f32 %f3, %f3, %s_sign;
            setp.ne.u32 %p1, %s_interleaved, 0;
            shl.b32 %r10, %r6, 1;
            add.u32 %r11, %r10, 1;
            add.u32 %r13, %r6, %s_half;
            selp.u32 %r14, %r10, %r6, %p1;
            selp.u32 %r15, %r11, %r13, %p1;
            mul.lo.u32 %r16, %r5, %s_dim;
            add.u32 %r14, %r14, %r16;
            add.u32 %r15, %r15, %r16;
            mul.wide.u32 %rd5, %r14, 4;
            mul.wide.u32 %rd6, %r15, 4;
            add.u64 %rd7, %b_x, %rd5;
            ld.global.f32 %f4, [%rd7];
            add.u64 %rd7, %b_x, %rd6;
            ld.global.f32 %f5, [%rd7];
            mul.f32 %f6, %f5, %f3;
            neg.f32 %f6, %f6;
            fma.rn.f32 %f6, %f4, %f2, %f6;
            mul.f32 %f7, %f4, %f3;
            fma.rn.f32 %f7, %f5, %f2, %f7;
            add.u64 %rd8, %b_y, %rd5;
            st.global.f32 [%rd8], %f6;
            add.u64 %rd8, %b_y, %rd6;
            st.global.f32 [%rd8], %f7;
            """);
    }
}
