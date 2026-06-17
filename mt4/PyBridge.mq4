//+------------------------------------------------------------------+
//|  PyBridge.mq4 — file-based Python bridge for MT4 (Wine-safe)      |
//|                                                                  |
//|  Exchanges plain key=value files with src/core/mt4_connector.py  |
//|  under  <terminal-data>/MQL4/Files/pybridge/ :                   |
//|    cmd/<id>.req   commands written by Python (atomic)            |
//|    res/<id>.res   results written here (terminated by eof=1)     |
//|    res/<id>.csv   rate data for 'rates' requests                |
//|    account.txt    account snapshot (refreshed every timer)      |
//|    positions.csv  open positions snapshot                       |
//|                                                                  |
//|  No DLLs / no DLL imports — only "Allow automated trading".      |
//|  Attach to ONE chart of any symbol.                              |
//+------------------------------------------------------------------+
#property strict

input int    TimerMs      = 100;     // poll/snapshot interval (ms)
input int    Slippage     = 20;      // points
input string BridgeDir    = "pybridge";

string CMD, RES;

//+------------------------------------------------------------------+
int OnInit()
  {
   CMD = BridgeDir + "\\cmd\\";
   RES = BridgeDir + "\\res\\";
   // FileWrite auto-creates subfolders; touch the snapshot so Python sees us "alive"
   WriteAccount();
   WritePositions();
   EventSetMillisecondTimer(TimerMs);
   Print("PyBridge started — dir=", BridgeDir, "  files=", TerminalInfoString(TERMINAL_DATA_PATH));
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTick() { }   // trading driven by timer/commands, not ticks

//+------------------------------------------------------------------+
void OnTimer()
  {
   ProcessCommands();
   WriteAccount();
   WritePositions();
  }

//+------------------------------------------------------------------+
//| Command dispatch                                                 |
//+------------------------------------------------------------------+
void ProcessCommands()
  {
   string fname;
   long h = FileFindFirst(CMD + "*.req", fname);
   if(h == INVALID_HANDLE) return;
   do
     {
      string path = CMD + fname;
      string keys[], vals[];
      int n = ReadKV(path, keys, vals);
      FileDelete(path);
      if(n <= 0) continue;
      string id     = KV(keys, vals, n, "id");
      string action = KV(keys, vals, n, "action");
      if(id == "") continue;
      Dispatch(id, action, keys, vals, n);
     }
   while(FileFindNext(h, fname));
   FileFindClose(h);
  }

void Dispatch(string id, string action, string &keys[], string &vals[], int n)
  {
   if(action == "ping")        { ResOK(id, ""); return; }
   if(action == "tick")        { DoTick(id, KV(keys,vals,n,"symbol")); return; }
   if(action == "symbol")      { DoSymbol(id, KV(keys,vals,n,"symbol")); return; }
   if(action == "rates")       { DoRates(id, KV(keys,vals,n,"symbol"), KV(keys,vals,n,"tf"), (int)StringToInteger(KV(keys,vals,n,"count"))); return; }
   if(action == "open")        { DoOpen(id, keys, vals, n); return; }
   if(action == "close")       { DoClose(id, (int)StringToInteger(KV(keys,vals,n,"ticket")), StringToDouble(KV(keys,vals,n,"volume"))); return; }
   if(action == "modify")      { DoModify(id, (int)StringToInteger(KV(keys,vals,n,"ticket")), StringToDouble(KV(keys,vals,n,"sl")), StringToDouble(KV(keys,vals,n,"tp"))); return; }
   ResErr(id, "unknown action: " + action);
  }

//+------------------------------------------------------------------+
//| Actions                                                          |
//+------------------------------------------------------------------+
void DoTick(string id, string sym)
  {
   if(sym != "") SymbolSelect(sym, true);
   double ask = MarketInfo(sym, MODE_ASK);
   double bid = MarketInfo(sym, MODE_BID);
   if(ask <= 0 && bid <= 0) { ResErr(id, "no tick for " + sym); return; }
   string b = "ok=1\n";
   b += "ask=" + DoubleToString(ask, 8) + "\n";
   b += "bid=" + DoubleToString(bid, 8) + "\n";
   b += "last=" + DoubleToString(bid, 8) + "\n";
   b += "time=" + IntegerToString(TimeCurrent()) + "\n";
   WriteRes(id, b);
  }

void DoSymbol(string id, string sym)
  {
   if(sym != "") SymbolSelect(sym, true);
   string b = "ok=1\n";
   b += "point="        + DoubleToString(MarketInfo(sym, MODE_POINT), 10) + "\n";
   b += "tick_value="   + DoubleToString(MarketInfo(sym, MODE_TICKVALUE), 10) + "\n";
   b += "tick_size="    + DoubleToString(MarketInfo(sym, MODE_TICKSIZE), 10) + "\n";
   b += "volume_min="   + DoubleToString(MarketInfo(sym, MODE_MINLOT), 4) + "\n";
   b += "volume_max="   + DoubleToString(MarketInfo(sym, MODE_MAXLOT), 4) + "\n";
   b += "volume_step="  + DoubleToString(MarketInfo(sym, MODE_LOTSTEP), 4) + "\n";
   b += "digits="       + IntegerToString((int)MarketInfo(sym, MODE_DIGITS)) + "\n";
   WriteRes(id, b);
  }

void DoRates(string id, string sym, string tf, int count)
  {
   if(sym != "") SymbolSelect(sym, true);
   if(count <= 0) count = 500;
   int period = TfToPeriod(tf);
   MqlRates r[];
   ArraySetAsSeries(r, true);
   int got = CopyRates(sym, period, 0, count, r);
   if(got <= 0) { ResErr(id, "no rates for " + sym + " " + tf); return; }
   // write CSV oldest→newest (match MT5 copy_rates_from_pos order)
   string csv = RES + id + ".csv";
   int fh = FileOpen(csv, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE) { ResErr(id, "cannot open csv"); return; }
   FileWriteString(fh, "time,open,high,low,close,tick_volume,spread,real_volume\n");
   for(int i = got - 1; i >= 0; i--)
     {
      string line = IntegerToString(r[i].time) + ","
                  + DoubleToString(r[i].open, 8) + ","
                  + DoubleToString(r[i].high, 8) + ","
                  + DoubleToString(r[i].low, 8) + ","
                  + DoubleToString(r[i].close, 8) + ","
                  + IntegerToString(r[i].tick_volume) + ",0,0\n";
      FileWriteString(fh, line);
     }
   FileClose(fh);
   string b = "ok=1\nrows=" + IntegerToString(got) + "\nfile=" + id + ".csv\n";
   WriteRes(id, b);
  }

void DoOpen(string id, string &keys[], string &vals[], int n)
  {
   string sym  = KV(keys, vals, n, "symbol");
   string type = KV(keys, vals, n, "type");
   double vol  = StringToDouble(KV(keys, vals, n, "volume"));
   double sl   = StringToDouble(KV(keys, vals, n, "sl"));
   double tp   = StringToDouble(KV(keys, vals, n, "tp"));
   int    mg   = (int)StringToInteger(KV(keys, vals, n, "magic"));
   string cmt  = KV(keys, vals, n, "comment");
   if(sym != "") SymbolSelect(sym, true);

   int    op    = (type == "buy") ? OP_BUY : OP_SELL;
   double price = (type == "buy") ? MarketInfo(sym, MODE_ASK) : MarketInfo(sym, MODE_BID);
   int    dig   = (int)MarketInfo(sym, MODE_DIGITS);
   price = NormalizeDouble(price, dig);

   int ticket = OrderSend(sym, op, vol, price, Slippage, NormalizeDouble(sl, dig),
                          NormalizeDouble(tp, dig), cmt, mg, 0, clrNONE);
   if(ticket < 0) { ResErr(id, "OrderSend err " + IntegerToString(GetLastError())); return; }
   string b = "ok=1\nretcode=10009\nticket=" + IntegerToString(ticket)
            + "\nprice=" + DoubleToString(price, dig) + "\n";
   WriteRes(id, b);
  }

void DoClose(string id, int ticket, double volume)
  {
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) { ResErr(id, "no ticket " + IntegerToString(ticket)); return; }
   string sym = OrderSymbol();
   int    dig = (int)MarketInfo(sym, MODE_DIGITS);
   double lots = (volume > 0 && volume < OrderLots()) ? volume : OrderLots();
   double price = (OrderType() == OP_BUY) ? MarketInfo(sym, MODE_BID) : MarketInfo(sym, MODE_ASK);
   price = NormalizeDouble(price, dig);
   if(!OrderClose(ticket, lots, price, Slippage, clrNONE)) { ResErr(id, "OrderClose err " + IntegerToString(GetLastError())); return; }
   string b = "ok=1\nretcode=10009\nprice=" + DoubleToString(price, dig) + "\n";
   WriteRes(id, b);
  }

void DoModify(string id, int ticket, double sl, double tp)
  {
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) { ResErr(id, "no ticket " + IntegerToString(ticket)); return; }
   string sym = OrderSymbol();
   int    dig = (int)MarketInfo(sym, MODE_DIGITS);
   if(!OrderModify(ticket, OrderOpenPrice(), NormalizeDouble(sl, dig), NormalizeDouble(tp, dig), 0, clrNONE))
     { ResErr(id, "OrderModify err " + IntegerToString(GetLastError())); return; }
   WriteRes(id, "ok=1\nretcode=10009\n");
  }

//+------------------------------------------------------------------+
//| Snapshots                                                        |
//+------------------------------------------------------------------+
void WriteAccount()
  {
   string b = "";
   b += "login="       + IntegerToString(AccountNumber()) + "\n";
   b += "balance="     + DoubleToString(AccountBalance(), 2) + "\n";
   b += "equity="      + DoubleToString(AccountEquity(), 2) + "\n";
   b += "margin_free=" + DoubleToString(AccountFreeMargin(), 2) + "\n";
   b += "currency="    + AccountCurrency() + "\n";
   b += "server="      + AccountServer() + "\n";
   b += "ts="          + IntegerToString(TimeCurrent()) + "\n";
   WriteAtomic(BridgeDir + "\\account.txt", b);
  }

void WritePositions()
  {
   string b = "ticket,symbol,type,volume,price_open,price_current,sl,tp,profit,magic,comment\n";
   for(int i = 0; i < OrdersTotal(); i++)
     {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;   // skip pending
      string sym = OrderSymbol();
      double cur = (OrderType() == OP_BUY) ? MarketInfo(sym, MODE_BID) : MarketInfo(sym, MODE_ASK);
      b += IntegerToString(OrderTicket()) + "," + sym + "," + IntegerToString(OrderType()) + ","
         + DoubleToString(OrderLots(), 2) + "," + DoubleToString(OrderOpenPrice(), 8) + ","
         + DoubleToString(cur, 8) + "," + DoubleToString(OrderStopLoss(), 8) + ","
         + DoubleToString(OrderTakeProfit(), 8) + "," + DoubleToString(OrderProfit() + OrderSwap() + OrderCommission(), 2) + ","
         + IntegerToString(OrderMagicNumber()) + "," + OrderComment() + "\n";
     }
   WriteAtomic(BridgeDir + "\\positions.csv", b);
  }

//+------------------------------------------------------------------+
//| File helpers                                                     |
//+------------------------------------------------------------------+
void WriteRes(string id, string body)   // results terminated by eof=1 so Python reads only complete files
  {
   WriteAtomic(RES + id + ".res", body + "id=" + id + "\neof=1\n");
  }
void ResOK(string id, string extra)  { WriteRes(id, "ok=1\n" + extra); }
void ResErr(string id, string msg)   { WriteRes(id, "ok=0\nerror=" + msg + "\n"); }

void WriteAtomic(string path, string body)
  {
   int fh = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE) return;
   FileWriteString(fh, body);
   FileClose(fh);
  }

int ReadKV(string path, string &keys[], string &vals[])
  {
   int fh = FileOpen(path, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE) return(0);
   int n = 0;
   ArrayResize(keys, 0); ArrayResize(vals, 0);
   while(!FileIsEnding(fh))
     {
      string line = FileReadString(fh);
      int eq = StringFind(line, "=");
      if(eq <= 0) continue;
      ArrayResize(keys, n + 1); ArrayResize(vals, n + 1);
      keys[n] = StringTrimRight(StringTrimLeft(StringSubstr(line, 0, eq)));
      vals[n] = StringTrimRight(StringTrimLeft(StringSubstr(line, eq + 1)));
      n++;
     }
   FileClose(fh);
   return(n);
  }

string KV(string &keys[], string &vals[], int n, string key)
  {
   for(int i = 0; i < n; i++) if(keys[i] == key) return(vals[i]);
   return("");
  }

int TfToPeriod(string tf)
  {
   tf = StringToUpper(tf);
   if(tf == "M1")  return(PERIOD_M1);
   if(tf == "M5")  return(PERIOD_M5);
   if(tf == "M15") return(PERIOD_M15);
   if(tf == "M30") return(PERIOD_M30);
   if(tf == "H1")  return(PERIOD_H1);
   if(tf == "H4")  return(PERIOD_H4);
   if(tf == "D1")  return(PERIOD_D1);
   if(tf == "W1")  return(PERIOD_W1);
   if(tf == "MN1") return(PERIOD_MN1);
   return(PERIOD_M15);
  }
//+------------------------------------------------------------------+
