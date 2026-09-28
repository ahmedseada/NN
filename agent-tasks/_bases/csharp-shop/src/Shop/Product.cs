namespace Shop;

public sealed record Product(string Sku, string Name, Money Price, string Category);
