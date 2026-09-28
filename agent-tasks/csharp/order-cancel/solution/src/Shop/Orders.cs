namespace Shop;

public enum OrderStatus
{
    Placed,
    Cancelled,
}

public sealed record Order(Guid Id, IReadOnlyList<CartLine> Lines, Money Total, DateTimeOffset PlacedAt)
{
    public OrderStatus Status { get; init; } = OrderStatus.Placed;
}

public interface IOrderRepository
{
    Task SaveAsync(Order order, CancellationToken cancellationToken = default);

    Task<Order?> FindAsync(Guid id, CancellationToken cancellationToken = default);

    Task UpdateAsync(Order order, CancellationToken cancellationToken = default);
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

        foreach (var line in cart.Lines)
        {
            inventory.Remove(line.Product.Sku, line.Quantity);
        }

        var order = new Order(Guid.NewGuid(), [.. cart.Lines], cart.Subtotal(), clock.GetUtcNow());
        await orders.SaveAsync(order, cancellationToken);
        return order;
    }

    public async Task<Order> CancelAsync(Guid orderId, CancellationToken cancellationToken = default)
    {
        var order = await orders.FindAsync(orderId, cancellationToken) ?? throw new KeyNotFoundException($"No order {orderId}.");
        if (order.Status == OrderStatus.Cancelled)
        {
            throw new InvalidOperationException($"Order {orderId} is already cancelled.");
        }

        foreach (var line in order.Lines)
        {
            inventory.AddStock(line.Product.Sku, line.Quantity);
        }

        var cancelled = order with { Status = OrderStatus.Cancelled };
        await orders.UpdateAsync(cancelled, cancellationToken);
        return cancelled;
    }
}
