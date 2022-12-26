
from BitMex_Conn import BitMEXWebsocket
import logging
from time import sleep


def run():
    logger = setup_logger()
    #instantiate the ws
    ws = BitMEXWebsocket(endpoint = "wss://ws.testnet.bitmex.com/realtime", symbol = "XBTUSD",
    api_key = None, api_secret = None)

    #print("\nHere is the default subscription\n**{}**\n\n\nDone...".format(ws.generate_signature))


    logger.info("Instrument data: {}".format(ws.get_instrument()))
        #logs the instrument data: instrument data is things like the symbol, state, when it was listed, etc
            #max order quantity, makerFee and takerFee

    #run time
    while(ws.ws.sock.connected):
        #while connected to web sockets
        logger.info("Ticker: %s" % ws.get_ticker())
            #contains 'last', 'buy', 'sell', 'mid' prices
        if ws.api_key:
            logger.info("Funds: %s" % ws.funds())
        #logger.info("Market Depth: %s" % ws.market_depth())
        ws.store_depth(ws.market_depth())
            #a list of dicts of orders. with the side (sell or buy) and size and price of the order. sells first.
        #logger.info("Recent Trades: %s\n\n" % ws.recent_trades())
            # i think this is just the most recent trade

        sleep(10)

def setup_logger():
    #prints logger info
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Change this to DEBUG if you want a lot more info
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    run()
