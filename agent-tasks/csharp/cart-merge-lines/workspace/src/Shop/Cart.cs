namespace Shop;

public sealed record CartLine(Product Product, int Quantity)
{
    public Money Total => Product.Price.Multiply(Quantity);
}

/// <summary>A shopping cart in one currency.</summary>
public sealed class Cart(string currency)
{
    private readonly List<CartLine> _lines = [];

    public string Currency { get; } = currency;

    public IReadOnlyList<CartLine> Lines => _lines;

    public void Add(Product product, int quantity = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        if (product.Price.Currency != Currency)
        {
            throw new InvalidOperationException($"The cart is in {Currency}, {product.Sku} is priced in {product.Price.Currency}.");
        }

        _lines.Add(new CartLine(product, quantity));
    }

    public bool Remove(string sku) => _lines.RemoveAll(l => l.Product.Sku == sku) > 0;

    public Money Subtotal() => _lines.Aggregate(Money.Zero(Currency), (sum, line) => sum.Add(line.Total));
}
