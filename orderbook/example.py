from orderbook import OrderBook

# Create an order book

order_book = OrderBook()

# Create some limit orders

limit_orders = [
                {'type' : 'limit',  'side' : 'bid', 'quantity' : 200,'price' : 161, 'trade_id' : 100, 'order_id' : 100, 'timestamp' : 1},
                {'type' : 'limit',  'side' : 'bid', 'quantity' : 300,'price' : 162, 'trade_id' : 101, 'order_id' : 101, 'timestamp' : 2},
                {'type' : 'limit',  'side' : 'bid', 'quantity' : 500,'price' : 163, 'trade_id' : 102, 'order_id' : 102, 'timestamp' : 3},
                {'type' : 'limit',  'side' : 'bid', 'quantity' : 200,'price' : 164, 'trade_id' : 103, 'order_id' : 103, 'timestamp' : 4},
                {'type' : 'limit',  'side' : 'ask', 'quantity' : 400,'price' : 160, 'trade_id' : 200, 'order_id' : 200, 'timestamp' : 5},
                {'type' : 'limit',  'side' : 'ask', 'quantity' : 400,'price' : 161, 'trade_id' : 201, 'order_id' : 201, 'timestamp' : 6},
                {'type' : 'limit',  'side' : 'ask', 'quantity' : 100,'price' : 162, 'trade_id' : 202, 'order_id' : 202, 'timestamp' : 7},
                {'type' : 'limit',  'side' : 'ask', 'quantity' : 300,'price' : 163, 'trade_id' : 203, 'order_id' : 203, 'timestamp' : 8},
                {'type' : 'limit',  'side' : 'ask', 'quantity' : 300,'price' : 164, 'trade_id' : 204, 'order_id' : 204, 'timestamp' : 9},
               ]

# Add orders to order book
for order in limit_orders:
    trades, order_id = order_book.process_order(order, True, False)
    #print(trades, order_id)


# The current book may be viewed using a print
print(order_book)


