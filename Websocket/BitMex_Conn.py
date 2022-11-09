
#unauthenticated as we don't need any user data

import websocket
import threading
import traceback
from datetime import datetime
from time import sleep
import json
import logging
import urllib.parse
import math
from util.subscriptions import NO_SYMBOL_SUBS, DEFAULT_SUBS
from util.api_key import generate_nonce, generate_signature
import csv
#We want to stream data from the API instead of polling the API constantly

class BitMEXWebsocket:
    #streaming connection to BitMEXWebsocket

    MAX_TABLE_LEN = 200

    def __init__(self, endpoint, symbol, api_key = None, api_secret = None, subscriptions = DEFAULT_SUBS):
        '''connect to the web socket and initialize data'''
        print("\nHere is the default subscription\n**{}**\n\n\nDone".format(DEFAULT_SUBS)) #this print will never activate. Maybe the initializer happens internally only
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing WebSocket.")

        self.endpoint = endpoint #assign endpoint from API
        self.symbol = symbol #assign symbol
        self.subscriptions = subscriptions

        if api_key is not None and api_secret is None:
            raise ValueError('api_secret required when api_key is provided')
        if api_key is None and api_secret is not None:
            raise ValueError('api_key required when api_secret is provided')

        self.api_key = api_key #defaulted to none
        self.api_secret = api_secret

        self.data = {}
        self.keys = {}
        self.exited = False

        #subscribe to endpoints
        wsURL = self.__get_url(subscriptions)
            #
        self.logger.info("Connecting to {}".format(wsURL))
        self.__connect(wsURL, symbol)
            #connect to web socket url and the desired symbol. ex:XBTUSD
        self.logger.info('Connected to WS.')

        #Now that we connected to endpoints, wait for the partials
            #a partial is like a SQL select, where you get just the data you want. in this case, just the symbol
        self.__wait_for_symbol(symbol)
        if api_key: #if api key is present
            self.__wait_for_account()
        self.logger.info('Market data retrieved')

    def exit(self):
        ''' close websocket '''
        self.exited = True
        self.ws.close()

#relevant instrument data
    #symbol, rootSymbol, positionCurrency,lotSize, tickSize, makerFee, takerFee, volume(tick volume), volume24h
    #lastTickDirection, impactBidPrice, impactMidPrice, impactAskPrice, openInterest, openValue
        #fairbasisrate, fairbasis, fairprice for if I want to work with futures and such
                #open interst , is this the same as number of orders in market depth?

    def get_instrument(self):
        '''Get instrument data for the symbol'''
        instrument = self.data['instrument'][0]
        instrument['tickLog'] = int(math.fabs(math.log10(instrument['tickSize'])))
        return instrument

#From the Ticker
    #last, buy, sell, mid
    def get_ticker(self):
        '''Return the ticker object that we want'''
        lastQuote = self.data['quote'][-1]
        lastTrade = self.data['trade'][-1]
        ticker = {
            "last" : lastTrade['price'],
            "buy" : lastQuote['bidPrice'],
            "sell": lastQuote['askPrice'],
            "mid": (float(lastQuote['bidPrice'] or 0) + float(lastQuote['askPrice'] or 0)) / 2
        }

        instrument = self.data['instrument'][0]
        return {k: round(float(v or 0), instrument['tickLog']) for k, v in ticker.items()}

    def store_ticker(self):
        

    def funds(self):
        '''Get your margin details.'''
        return self.data['margin'][0]

    def positions(self):
        '''Get your positions.'''
        return self.data['position']

#symbol, id, side, size, price, timestamp

    def market_depth(self):
        '''Get market depth (orderbook). Returns all levels.'''
        return self.data['orderBookL2']

    def open_orders(self, clOrdIDPrefix):
        '''Get all your open orders.'''
        orders = self.data['order']
        # Filter to only open orders and those that we actually placed
        return [o for o in orders if str(o['clOrdID']).startswith(clOrdIDPrefix) and order_leaves_quantity(o)]

    def recent_trades(self):
        '''Get recent trades.'''
        return self.data['trade']

    def store_depth(self, orderBook):
        '''Collect buy and sell data from L2 order book, write to csv'''
        sells = {} #dictionary of prices with number of sells at different prices
        buys = {}
        for order in orderBook:
            order_type = order['side']
            size = order['size']
            price = order['price']
            timestamp = order['timestamp']

            if order_type == 'Sell':
                sells[price] = (size, timestamp)
            else:
                buys[price] = (size, timestamp)
        current_time = datetime.now()
        csv_file = str(current_time.strftime("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/Data/%m_%d_%Y,%H_%M_%S") + 'sells.csv')

        with open(csv_file, 'w') as f:
            w = csv.writer(f, delimiter = ',')
            for key, value in sorted(sells.items()):
                order_size, order_time = value
                w.writerow([key, order_size, order_time)

        csv_file = str(current_time.strftime("C:/Users/brand/OneDrive/Documents/Python/PyOrderBook/Data/%m_%d_%Y,%H_%M_%S") + 'buys.csv')

        with open(csv_file, 'w') as f:
            w = csv.writer(f, delimiter = ',')
            for key, value in sorted(buys.items()):
                order_size, order_time = value
                w.writerow([key, order_size, order_time])



    #
    # End Public Methods
    #

    def __connect(self, wsURL, symbol):
        '''Connect to the websocket in a thread.'''
        self.logger.debug("Starting thread")

        self.ws = websocket.WebSocketApp(wsURL,
                                         on_message=self.__on_message,
                                         on_close=self.__on_close,
                                         on_open=self.__on_open,
                                         on_error=self.__on_error,
                                         header=self.__get_auth())

        self.wst = threading.Thread(target=lambda: self.ws.run_forever())
        self.wst.daemon = True
        self.wst.start()
        self.logger.debug("Started thread")

        # Wait for connect before continuing
        conn_timeout = 5
        while (not self.ws.sock or not self.ws.sock.connected) and conn_timeout:
            sleep(1)
            conn_timeout -= 1
        if not conn_timeout:
            self.logger.error("Couldn't connect to WS! Exiting.")
            self.exit()
            raise websocket.WebSocketTimeoutException('Couldn\'t connect to WS! Exiting.')

    def __get_auth(self):
        '''Return auth headers. Will use API Keys if present in settings.'''
        if self.api_key:
            self.logger.info("Authenticating with API Key.")
            # To auth to the WS using an API key, we generate a signature of a nonce and
            # the WS API endpoint.
            expires = generate_nonce()
            return [
                "api-expires: " + str(expires),
                "api-signature: " + generate_signature(self.api_secret, 'GET', '/realtime', expires, ''),
                "api-key:" + self.api_key
            ]
        else:
            self.logger.info("Not authenticating.")
            return []

    def __get_url(self, subscriptions):
        '''
        Generate a connection URL. We can define subscriptions right in the querystring.
        Most subscription topics are scoped by the symbol we're listening to.
        '''

        # Some subscriptions need to have the symbol appended.
        subscriptions_full = map(lambda sub: (
            sub if sub in NO_SYMBOL_SUBS
            else (sub + ':' + self.symbol)
        ), subscriptions)

        urlParts = list(urllib.parse.urlparse(self.endpoint))
        urlParts[2] += "?subscribe={}".format(','.join(subscriptions_full))
        return urllib.parse.urlunparse(urlParts)

    def __wait_for_account(self):
        '''On subscribe, this data will come down. Wait for it.'''
        # Wait for the keys to show up from the ws
        while not {'margin', 'position', 'order', 'orderBookL2'} <= set(self.data):
            sleep(0.1)

    def __wait_for_symbol(self, symbol):
        '''On subscribe, this data will come down. Wait for it.'''
        while not {'instrument', 'trade', 'quote'} <= set(self.data):
            sleep(0.1)

    def __send_command(self, command, args=None):
        '''Send a raw command.'''
        if args is None:
            args = []
        self.ws.send(json.dumps({"op": command, "args": args}))

    def __on_message(self, message):
        '''Handler for parsing JSON WS messages.'''
        message = json.loads(message) # turns into a panda dictionary
        self.logger.debug(json.dumps(message))

        table = message.get("table")
        action = message.get("action")
        try:
            if 'subscribe' in message:
                #if we subscribed properly to an endpoint already
                self.logger.debug("Subscribed to %s." % message['subscribe'])
            elif action:
                #if not subscribed,
                if table not in self.data:
                    #if the table is not in the data already
                    self.data[table] = []

                # There are four possible actions from the WS:
                # 'partial' - full table image
                # 'insert'  - new row
                # 'update'  - update row
                # 'delete'  - delete row
                if action == 'partial':
                    self.logger.debug("%s: partial" % table)
                    self.data[table] = message['data']
                    # Keys are communicated on partials to let you know how to uniquely identify
                    # an item. We use it for updates.
                    self.keys[table] = message['keys']
                elif action == 'insert':
                    self.logger.debug('%s: inserting %s' % (table, message['data']))
                    self.data[table] += message['data']

                    # Limit the max length of the table to avoid excessive memory usage.
                    # Don't trim orders because we'll lose valuable state if we do.
                    if table not in ['order', 'orderBookL2'] and len(self.data[table]) > BitMEXWebsocket.MAX_TABLE_LEN:
                        self.data[table] = self.data[table][BitMEXWebsocket.MAX_TABLE_LEN // 2:]

                elif action == 'update':
                    self.logger.debug('%s: updating %s' % (table, message['data']))
                    # Locate the item in the collection and update it.
                    for updateData in message['data']:
                        item = find_by_keys(self.keys[table], self.data[table], updateData)
                        if not item:
                            return  # No item found to update. Could happen before push
                        item.update(updateData)
                        # Remove cancelled / filled orders
                        if table == 'order' and not order_leaves_quantity(item):
                            self.data[table].remove(item)
                elif action == 'delete':
                    self.logger.debug('%s: deleting %s' % (table, message['data']))
                    # Locate the item in the collection and remove it.
                    for deleteData in message['data']:
                        item = find_by_keys(self.keys[table], self.data[table], deleteData)
                        self.data[table].remove(item)
                else:
                    raise Exception("Unknown action: %s" % action)
        except:
            self.logger.error(traceback.format_exc())

    def __on_error(self, error):
        '''Called on fatal websocket errors. We exit on these.'''
        if not self.exited:
            self.logger.error("Error : %s" % error)
            raise websocket.WebSocketException(error)

    def __on_open(self):
        '''Called when the WS opens.'''
        self.logger.debug("Websocket Opened.")

    def __on_close(self):
        '''Called on websocket close.'''
        self.logger.info('Websocket Closed')


# Utility method for finding an item in the store.
# When an update comes through on the websocket, we need to figure out which item in the array it is
# in order to match that item.
#
# Helpfully, on a data push (or on an HTTP hit to /api/v1/schema), we have a "keys" array. These are the
# fields we can use to uniquely identify an item. Sometimes there is more than one, so we iterate through all
# provided keys.
def find_by_keys(keys, table, matchData):
    for item in table:
        if all(item[k] == matchData[k] for k in keys):
            return item

def order_leaves_quantity(o):
    if o['leavesQty'] is None:
        return True
    return o['leavesQty'] > 0
