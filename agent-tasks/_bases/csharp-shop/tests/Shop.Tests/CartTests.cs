namespace Shop.Tests;

public class CartTests
{
    private static readonly Product Mug = new("MUG-1", "Mug", new Money(8.50m, "EUR"), "Kitchen");

    [Fact]
    public void SubtotalSumsLines()
    {
        var cart = new Cart("EUR");
        cart.Add(Mug, 2);
        Assert.Equal(new Money(17m, "EUR"), cart.Subtotal());
    }

    [Fact]
    public void RefusesProductsInAnotherCurrency()
    {
        var cart = new Cart("USD");
        Assert.Throws<InvalidOperationException>(() => cart.Add(Mug));
    }
}
