namespace Shop.Tests;

public class OrderStockHiddenTests
{
    private static readonly Product Mug = new("MUG-1", "Mug", new Money(8.50m, "EUR"), "Kitchen");
    private static readonly Product Pan = new("PAN-2", "Pan", new Money(30m, "EUR"), "Kitchen");

    [Fact]
    public async Task AnOrderThatCannotBeFilledChangesNothing()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 5);
        inventory.AddStock("PAN-2", 1);
        var repository = new OrderServiceTests.MemoryOrders();
        var cart = new Cart("EUR");
        cart.Add(Mug, 2);
        cart.Add(Pan, 2);

        await Assert.ThrowsAsync<InvalidOperationException>(() => new OrderService(inventory, repository, TimeProvider.System).PlaceOrderAsync(cart));

        Assert.Equal(5, inventory.Quantity("MUG-1"));
        Assert.Equal(1, inventory.Quantity("PAN-2"));
        Assert.Equal(0, repository.Saved);
    }

    [Fact]
    public async Task AProductNeverStockedCannotBeOrdered()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 5);
        var cart = new Cart("EUR");
        cart.Add(Mug);
        cart.Add(Pan);

        await Assert.ThrowsAnyAsync<Exception>(() => new OrderService(inventory, new OrderServiceTests.MemoryOrders(), TimeProvider.System).PlaceOrderAsync(cart));
        Assert.Equal(5, inventory.Quantity("MUG-1"));
    }
}
