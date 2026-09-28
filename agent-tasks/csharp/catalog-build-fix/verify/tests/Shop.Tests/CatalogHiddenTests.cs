namespace Shop.Tests;

public class CatalogHiddenTests
{
    private static readonly Catalog Shop = new([
        new Product("MUG-1", "Blue Mug", new Money(8.50m, "EUR"), "Kitchen"),
        new Product("MUG-2", "Large mug", new Money(6m, "EUR"), "Kitchen"),
        new Product("LAMP-1", "Desk lamp", new Money(25m, "EUR"), "Lighting"),
    ]);

    [Fact]
    public void SearchesNamesIgnoringCaseCheapestFirst() =>
        Assert.Equal(["MUG-2", "MUG-1"], Shop.Search("MUG").Select(p => p.Sku));

    [Fact]
    public void FindsBySkuOrReturnsNull()
    {
        Assert.Equal("Desk lamp", Shop.Find("lamp-1")?.Name);
        Assert.Null(Shop.Find("NOPE"));
    }

    [Fact]
    public void GroupsByCategory() =>
        Assert.Equal(["Kitchen", "Lighting"], Shop.ByCategory().Keys);

    [Fact]
    public void CheapestInACategory() => Assert.Equal(new Money(6m, "EUR"), Shop.CheapestIn("Kitchen"));
}
