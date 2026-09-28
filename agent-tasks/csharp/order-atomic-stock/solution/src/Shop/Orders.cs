namespace Shop;

public sealed record Order(Guid Id, IReadOnlyList<CartLine> Lines, Money Total, DateTimeOffset PlacedAt);

public interface IOrderRepository
{
    Task SaveAsync(Order order, CancellationToken cancellationToken = default);

    Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default);
}

/// <summary>Turns carts into orders.</summary>
public sealed class OrderService(Inventory inventory, IOrderRepository orders, TimeProvider clock)
{
    public async Task<Order> PlaceOrderAsync(Cart cart, CancellationToken cancellationToken = default)
    {
        if (cart.Lines.Count == 0)
        {
            throw new InvalidOperationException("The cart is empty.");
        }

        var missing = cart.Lines.FirstOrDefault(l => inventory.Quantity(l.Product.Sku) < l.Quantity);
        if (missing is not null)
        {
            throw new InvalidOperationException($"Not enough '{missing.Product.Sku}' in stock.");
        }

        foreach (var line in cart.Lines)
        {
            inventory.Remove(line.Product.Sku, line.Quantity);
        }

        var order = new Order(Guid.NewGuid(), [.. cart.Lines], cart.Subtotal(), clock.GetUtcNow());
        await orders.SaveAsync(order, cancellationToken);
        return order;
    }
}
