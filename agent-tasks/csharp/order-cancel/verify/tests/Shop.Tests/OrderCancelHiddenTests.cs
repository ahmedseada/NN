namespace Shop.Tests;

public class OrderCancelHiddenTests
{
    private static readonly Product Mug = new("MUG-1", "Mug", new Money(8.50m, "EUR"), "Kitchen");

    private sealed class Orders : IOrderRepository
    {
        public Dictionary<Guid, Order> Stored { get; } = [];

        public int Updates { get; private set; }

        public Task SaveAsync(Order order, CancellationToken cancellationToken = default)
        {
            Stored[order.Id] = order;
            return Task.CompletedTask;
        }

        public Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default) => Task.FromResult(Stored.GetValueOrDefault(id));

        public Task UpdateAsync(Order order, CancellationToken cancellationToken = default)
        {
            Updates++;
            Stored[order.Id] = order;
            return Task.CompletedTask;
        }
    }

    private static async Task<(OrderService Service, Inventory Inventory, Orders Orders, Order Order)> PlacedOrder()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 5);
        var orders = new Orders();
        var service = new OrderService(inventory, orders, TimeProvider.System);
        var cart = new Cart("EUR");
        cart.Add(Mug, 2);
        return (service, inventory, orders, await service.PlaceOrderAsync(cart));
    }

    [Fact]
    public async Task NewOrdersArePlaced() => Assert.Equal(OrderStatus.Placed, (await PlacedOrder()).Order.Status);

    [Fact]
    public async Task CancellingReturnsTheStockAndStoresTheStatus()
    {
        var (service, inventory, orders, order) = await PlacedOrder();
        var cancelled = await service.CancelAsync(order.Id);
        Assert.Equal(OrderStatus.Cancelled, cancelled.Status);
        Assert.Equal(OrderStatus.Cancelled, orders.Stored[order.Id].Status);
        Assert.Equal(5, inventory.Quantity("MUG-1"));
    }

    [Fact]
    public async Task CancellingTwiceFailsAndChangesNothing()
    {
        var (service, inventory, orders, order) = await PlacedOrder();
        await service.CancelAsync(order.Id);
        await Assert.ThrowsAsync<InvalidOperationException>(() => service.CancelAsync(order.Id));
        Assert.Equal(5, inventory.Quantity("MUG-1"));
        Assert.Equal(1, orders.Updates);
    }

    [Fact]
    public async Task CancellingAnUnknownOrderFails()
    {
        var (service, _, _, _) = await PlacedOrder();
        await Assert.ThrowsAsync<KeyNotFoundException>(() => service.CancelAsync(Guid.NewGuid()));
    }
}
