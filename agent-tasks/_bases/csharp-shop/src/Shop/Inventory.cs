namespace Shop;

/// <summary>Stock levels per SKU.</summary>
public sealed class Inventory
{
    private readonly Dictionary<string, int> _stock = new(StringComparer.OrdinalIgnoreCase);

    public void AddStock(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        _stock[sku] = Quantity(sku) + quantity;
    }

    public void Remove(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        if (!_stock.TryGetValue(sku, out int available))
        {
            throw new KeyNotFoundException($"Unknown SKU '{sku}'.");
        }

        if (available < quantity)
        {
            throw new InvalidOperationException($"Only {available} of '{sku}' in stock.");
        }

        _stock[sku] = available - quantity;
    }

    public int Quantity(string sku) => _stock.TryGetValue(sku, out int quantity) ? quantity : 0;

    public IReadOnlyDictionary<string, int> Snapshot() => new Dictionary<string, int>(_stock, StringComparer.OrdinalIgnoreCase);
}
