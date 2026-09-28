namespace Shop.Tests;

public class CartMergeHiddenTests
{
    private static readonly Product Mug = new("MUG-1", "Mug", new Money(8.50m, "EUR"), "Kitchen");
    private static readonly Product Pan = new("PAN-2", "Pan", new Money(30m, "EUR"), "Kitchen");

    [Fact]
    public void AddingTheSameProductAgainMergesTheLine()
    {
        var cart = new Cart("EUR");
        cart.Add(Mug, 2);
        cart.Add(Pan);
        cart.Add(Mug, 3);

        Assert.Equal(2, cart.Lines.Count);
        Assert.Equal(5, cart.Lines.Single(l => l.Product.Sku == "MUG-1").Quantity);
        Assert.Equal(new Money(72.50m, "EUR"), cart.Subtotal());
    }

    [Fact]
    public void RemoveDropsTheMergedLine()
    {
        var cart = new Cart("EUR");
        cart.Add(Mug);
        cart.Add(Mug);
        Assert.True(cart.Remove("MUG-1"));
        Assert.Empty(cart.Lines);
    }
}
