using NN.Backends;

namespace NN;

public sealed partial class Tensor
{
    // ---------------------------------------------------------------- operators

    /// <summary>Element-wise sum. A 1-D right operand whose length matches the last dimension is broadcast across rows (e.g. adding a bias).</summary>
    public static Tensor operator +(Tensor a, Tensor b) => Add(a, b);

    /// <summary>Element-wise difference of two tensors of the same shape.</summary>
    public static Tensor operator -(Tensor a, Tensor b) => ElementWise(BinaryOp.Sub, a, b);

    /// <summary>Element-wise (Hadamard) product of two tensors of the same shape. Use <see cref="MatMul"/> for matrix multiplication.</summary>
    public static Tensor operator *(Tensor a, Tensor b) => ElementWise(BinaryOp.Mul, a, b);

    /// <summary>Adds a constant to every element.</summary>
    public static Tensor operator +(Tensor a, float b) => a.Affine(1f, b);

    /// <summary>Adds a constant to every element.</summary>
    public static Tensor operator +(float a, Tensor b) => b.Affine(1f, a);

    /// <summary>Subtracts a constant from every element.</summary>
    public static Tensor operator -(Tensor a, float b) => a.Affine(1f, -b);

    /// <summary>Subtracts every element from a constant.</summary>
    public static Tensor operator -(float a, Tensor b) => b.Affine(-1f, a);

    /// <summary>Multiplies every element by a constant.</summary>
    public static Tensor operator *(Tensor a, float b) => a.Affine(b, 0f);

    /// <summary>Multiplies every element by a constant.</summary>
    public static Tensor operator *(float a, Tensor b) => b.Affine(a, 0f);

    /// <summary>Divides every element by a constant.</summary>
    public static Tensor operator /(Tensor a, float b) => a.Affine(1f / b, 0f);

    /// <summary>Negates every element.</summary>
    public static Tensor operator -(Tensor a) => a.Affine(-1f, 0f);

    // ---------------------------------------------------------------- activations

    /// <summary>Logistic sigmoid, 1 / (1 + e^-x), element-wise.</summary>
    public Tensor Sigmoid() => Unary(UnaryOp.Sigmoid);

    /// <summary>Hyperbolic tangent, element-wise.</summary>
    public Tensor Tanh() => Unary(UnaryOp.Tanh);

    /// <summary>Rectified linear unit, max(x, 0), element-wise.</summary>
    public Tensor Relu() => Unary(UnaryOp.Relu);

    /// <summary>x², element-wise.</summary>
    public Tensor Square() => Unary(UnaryOp.Square);

    // ---------------------------------------------------------------- linear algebra and reductions

    /// <summary>Matrix product of a [m, k] tensor with a [k, n] tensor, giving [m, n].</summary>
    public Tensor MatMul(Tensor other)
    {
        ThrowIfDisposed();
        other.ThrowIfDisposed();
        CheckSameDevice(this, other);
        if (Rank != 2 || other.Rank != 2 || _shape[1] != other._shape[0])
        {
            throw new ArgumentException($"MatMul needs shapes [m, k] and [k, n], got {FormatShape(_shape)} and {FormatShape(other._shape)}.");
        }

        int m = _shape[0], k = _shape[1], n = other._shape[1];
        var c = Empty([m, n], Device);
        Backend.MatMul(Storage, other.Storage, c.Storage, m, n, k, transA: false, transB: false, beta: 0f);

        if (WillRecord(this, other))
        {
            var a = this;
            var b = other;
            c.Record(g =>
            {
                if (a.RequiresGrad)
                {
                    // dA += dC · Bᵀ
                    a.Backend.MatMul(g.Storage, b.Storage, a.GradStorage(), m, k, n, transA: false, transB: true, beta: 1f);
                }

                if (b.RequiresGrad)
                {
                    // dB += Aᵀ · dC
                    b.Backend.MatMul(a.Storage, g.Storage, b.GradStorage(), k, n, m, transA: true, transB: false, beta: 1f);
                }
            }, a, b);
        }

        return c;
    }

    /// <summary>Sum of all elements, as a scalar tensor.</summary>
    public Tensor Sum() => Reduce(1f);

    /// <summary>Mean of all elements, as a scalar tensor.</summary>
    public Tensor Mean() => Reduce(Size == 0 ? 0f : 1f / Size);

    /// <summary>
    /// Returns a tensor with the same data viewed with a different shape. The data is shared, not copied.
    /// One dimension may be -1 to infer it from the others.
    /// </summary>
    public Tensor Reshape(params ReadOnlySpan<int> shape)
    {
        ThrowIfDisposed();
        var resolved = shape.ToArray();
        int inferred = Array.IndexOf(resolved, -1);
        if (inferred >= 0)
        {
            resolved[inferred] = 1;
            int known = ElementCount(resolved);
            resolved[inferred] = known == 0 ? 0 : Size / known;
        }

        if (ElementCount(resolved) != Size)
        {
            throw new ArgumentException($"Cannot reshape {FormatShape(_shape)} ({Size} elements) to {FormatShape(shape)}.");
        }

        Storage.AddRef();
        var y = new Tensor(resolved, Storage, Device, track: true);
        if (WillRecord(this))
        {
            var x = this;
            y.Record(g => x.Backend.Axpy(g.Storage, x.GradStorage(), x.Size, 1f), x);
        }

        return y;
    }

    // ---------------------------------------------------------------- implementations

    private Tensor Unary(UnaryOp op)
    {
        ThrowIfDisposed();
        var y = Empty(_shape, Device);
        Backend.Unary(op, Storage, y.Storage, Size);
        if (WillRecord(this))
        {
            var x = this;
            y.Record(g => x.Backend.UnaryBackward(op, x.Storage, y.Storage, g.Storage, x.GradStorage(), x.Size), x);
        }

        return y;
    }

    /// <summary>y = alpha * x + beta.</summary>
    private Tensor Affine(float alpha, float beta)
    {
        ThrowIfDisposed();
        var y = Empty(_shape, Device);
        Backend.Affine(Storage, y.Storage, Size, alpha, beta);
        if (WillRecord(this))
        {
            var x = this;
            y.Record(g => x.Backend.Axpy(g.Storage, x.GradStorage(), x.Size, alpha), x);
        }

        return y;
    }

    private Tensor Reduce(float scale)
    {
        ThrowIfDisposed();
        var y = Empty([], Device);
        Backend.Sum(Storage, y.Storage, Size, scale);
        if (WillRecord(this))
        {
            var x = this;
            y.Record(g => x.Backend.AddBroadcastScalar(g.Storage, x.GradStorage(), x.Size, scale), x);
        }

        return y;
    }

    private static Tensor Add(Tensor a, Tensor b)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        bool rowBroadcast = b.Rank == 1 && a.Rank >= 1 && a._shape[^1] == b._shape[0] && a.Rank != 1;
        if (!rowBroadcast)
        {
            return ElementWise(BinaryOp.Add, a, b);
        }

        CheckSameDevice(a, b);
        int cols = b.Size;
        int rows = cols == 0 ? 0 : a.Size / cols;
        var c = Empty(a._shape, a.Device);
        a.Backend.AddRowVector(a.Storage, b.Storage, c.Storage, rows, cols);
        if (WillRecord(a, b))
        {
            c.Record(g =>
            {
                if (a.RequiresGrad)
                {
                    a.Backend.Axpy(g.Storage, a.GradStorage(), a.Size, 1f);
                }

                if (b.RequiresGrad)
                {
                    b.Backend.SumRows(g.Storage, b.GradStorage(), rows, cols);
                }
            }, a, b);
        }

        return c;
    }

    private static Tensor ElementWise(BinaryOp op, Tensor a, Tensor b)
    {
        a.ThrowIfDisposed();
        b.ThrowIfDisposed();
        CheckSameDevice(a, b);
        CheckSameShape(a, b);
        var c = Empty(a._shape, a.Device);
        a.Backend.Binary(op, a.Storage, b.Storage, c.Storage, a.Size);
        if (WillRecord(a, b))
        {
            c.Record(g =>
            {
                var backend = a.Backend;
                int n = a.Size;
                switch (op)
                {
                    case BinaryOp.Add:
                        if (a.RequiresGrad) backend.Axpy(g.Storage, a.GradStorage(), n, 1f);
                        if (b.RequiresGrad) backend.Axpy(g.Storage, b.GradStorage(), n, 1f);
                        break;
                    case BinaryOp.Sub:
                        if (a.RequiresGrad) backend.Axpy(g.Storage, a.GradStorage(), n, 1f);
                        if (b.RequiresGrad) backend.Axpy(g.Storage, b.GradStorage(), n, -1f);
                        break;
                    case BinaryOp.Mul:
                        if (a.RequiresGrad) backend.MulAdd(g.Storage, b.Storage, a.GradStorage(), n);
                        if (b.RequiresGrad) backend.MulAdd(g.Storage, a.Storage, b.GradStorage(), n);
                        break;
                }
            }, a, b);
        }

        return c;
    }

    private static void CheckSameDevice(Tensor a, Tensor b)
    {
        if (a.Device != b.Device)
        {
            throw new InvalidOperationException($"Tensors are on different devices ({a.Device} and {b.Device}). Move one with .To(device).");
        }
    }

    private static void CheckSameShape(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException($"Shapes {FormatShape(a._shape)} and {FormatShape(b._shape)} do not match.");
        }
    }
}
