namespace Shop;

/// <summary>Stock levels per SKU, with stock held for pending orders.</summary>
public sealed class Inventory
{
    private readonly Dictionary<string, int> _stock = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, int> _reserved = new(StringComparer.OrdinalIgnoreCase);

    public void AddStock(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        _stock[sku] = Quantity(sku) + quantity;
    }

    public void Remove(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        if (!_stock.TryGetValue(sku, out int stock))
        {
            throw new KeyNotFoundException($"Unknown SKU '{sku}'.");
        }

        int available = Available(sku);
        if (available < quantity)
        {
            throw new InvalidOperationException($"Only {available} of '{sku}' available.");
        }

        _stock[sku] = stock - quantity;
    }

    public bool TryReserve(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        if (Available(sku) < quantity)
        {
            return false;
        }

        _reserved[sku] = Reserved(sku) + quantity;
        return true;
    }

    public void Release(string sku, int quantity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(quantity);
        int reserved = Reserved(sku);
        if (reserved < quantity)
        {
            throw new InvalidOperationException($"Only {reserved} of '{sku}' reserved.");
        }

        _reserved[sku] = reserved - quantity;
    }

    public int Quantity(string sku) => _stock.TryGetValue(sku, out int quantity) ? quantity : 0;

    public int Available(string sku) => Quantity(sku) - Reserved(sku);

    public IReadOnlyDictionary<string, int> Snapshot() => new Dictionary<string, int>(_stock, StringComparer.OrdinalIgnoreCase);

    private int Reserved(string sku) => _reserved.TryGetValue(sku, out int reserved) ? reserved : 0;
}
