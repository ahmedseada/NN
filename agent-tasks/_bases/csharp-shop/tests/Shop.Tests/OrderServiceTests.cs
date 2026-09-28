namespace Shop.Tests;

public class OrderServiceTests
{
    private static readonly Product Mug = new("MUG-1", "Mug", new Money(8.50m, "EUR"), "Kitchen");

    [Fact]
    public async Task PlacesAnOrderAndTakesTheStock()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 3);
        var repository = new MemoryOrders();
        var cart = new Cart("EUR");
        cart.Add(Mug, 2);

        var order = await new OrderService(inventory, repository, TimeProvider.System).PlaceOrderAsync(cart);

        Assert.Equal(new Money(17m, "EUR"), order.Total);
        Assert.Equal(1, inventory.Quantity("MUG-1"));
        Assert.Same(order, await repository.FindAsync(order.Id));
    }

    internal sealed class MemoryOrders : IOrderRepository
    {
        private readonly Dictionary<Guid, Order> _orders = [];

        public int Saved => _orders.Count;

        public Task SaveAsync(Order order, CancellationToken cancellationToken = default)
        {
            _orders[order.Id] = order;
            return Task.CompletedTask;
        }

        public Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default) =>
            Task.FromResult(_orders.GetValueOrDefault(id));
    }
}
