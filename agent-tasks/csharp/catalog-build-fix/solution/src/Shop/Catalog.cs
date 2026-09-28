namespace Shop;

/// <summary>The products on sale, searchable by name and grouped by category.</summary>
public sealed class Catalog(IEnumerable<Product> products)
{
    private readonly List<Product> _products = [.. products];

    /// <summary>Products whose name contains <paramref name="text"/> (case-insensitive), cheapest first.</summary>
    public IReadOnlyList<Product> Search(string text) =>
        _products.Where(p => p.Name.Contains(text, StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p.Price.Amount)
            .ToList();

    /// <summary>The product with <paramref name="sku"/>, or null.</summary>
    public Product? Find(string sku) => _products.FirstOrDefault(p => p.Sku.Equals(sku, StringComparison.OrdinalIgnoreCase));

    /// <summary>Category → products, categories in alphabetical order.</summary>
    public SortedDictionary<string, List<Product>> ByCategory()
    {
        var groups = new SortedDictionary<string, List<Product>>(StringComparer.Ordinal);
        foreach (var product in _products)
        {
            if (!groups.TryGetValue(product.Category, out var list))
            {
                list = new List<Product>();
                groups.Add(product.Category, list);
            }

            list.Add(product);
        }

        return groups;
    }

    public Money CheapestIn(string category) => ByCategory()[category].MinBy(p => p.Price.Amount)!.Price;
}
