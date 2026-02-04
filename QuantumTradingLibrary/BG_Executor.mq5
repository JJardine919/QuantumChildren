//+------------------------------------------------------------------+
//|                                              BG_Executor.mq5    |
//|                                  Copyright 2026, Quantum Library |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "Blue Guardian Sniper Executor"

//--- Input parameters
input string SignalFile = "etare_signals.json";      // Signal file from Python
input int    MagicNumber = 365060;                   // Blue Guardian Magic
input double LotSize = 2.5;                          // Fixed Lot Size
input int    Slippage = 50;                          // Max slippage
input bool   TradeEnabled = true;                    // ENABLED FOR LIVE TRADING

//--- Global variables
datetime lastSignalCheck = 0;
int checkInterval = 10;  // Check signals every 10 seconds

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("BLUE GUARDIAN EXECUTOR LOADED");
    Print("Account: ", AccountInfoInteger(ACCOUNT_LOGIN));
    Print("Lot Size: ", LotSize);
    Print("Trading Enabled: ", TradeEnabled);
    Print("========================================");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(TimeCurrent() - lastSignalCheck < checkInterval) return;
    lastSignalCheck = TimeCurrent();

    ReadAndExecuteSignals();
}

//+------------------------------------------------------------------+
//| Read signals and execute                                         |
//+------------------------------------------------------------------+
void ReadAndExecuteSignals()
{
    int fileHandle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_ANSI);
    if(fileHandle == INVALID_HANDLE) return;

    string json = "";
    while(!FileIsEnding(fileHandle)) json += FileReadString(fileHandle);
    FileClose(fileHandle);

    if(json == "") return;

    // Check for our symbol
    string currentSymbol = _Symbol;
    if(StringFind(json, "\"" + currentSymbol + "\"") < 0) return;

    // Parse Action
    string action = ExtractStringValue(json, "action");
    double confidence = ExtractDoubleValue(json, "confidence");

    if(!TradeEnabled) 
    {
        Print("DRY RUN >> Signal: ", action, " (", DoubleToString(confidence*100, 1), "%)");
        return;
    }

    // Execution Logic
    if(action == "BUY")
    {
        ExecuteTrade(ORDER_TYPE_BUY);
    }
    else if(action == "SELL")
    {
        ExecuteTrade(ORDER_TYPE_SELL);
    }
}

//+------------------------------------------------------------------+
//| Execute Trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type)
{
    // Check if position already exists for this magic
    if(PositionExists()) return;

    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // SL/TP (BTC Scale) - Hardcoded Safety
    double sl_dist = 500.0; // $500 move
    double tp_dist = 1500.0; // $1500 move
    
    double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist : price + sl_dist;
    double tp = (type == ORDER_TYPE_BUY) ? price + tp_dist : price - tp_dist;

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = LotSize;
    request.type = type;
    request.price = price;
    request.sl = NormalizeDouble(sl, _Digits);
    request.tp = NormalizeDouble(tp, _Digits);
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "BG_Redux";
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = ORDER_FILLING_IOC;

    if(!OrderSend(request, result))
    {
        Print("ERROR: OrderSend failed: ", GetLastError());
    }
    else
    {
        Print("SUCCESS: ", (type==ORDER_TYPE_BUY ? "BUY" : "SELL"), " executed.");
    }
}

bool PositionExists()
{
    for(int i=PositionsTotal()-1; i>=0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber) return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| JSON Helpers                                                     |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return "";
    int valueStart = StringFind(json, "\"", keyPos + StringLen(key) + 3);
    if(valueStart < 0) return "";
    int valueEnd = StringFind(json, "\"", valueStart + 1);
    return StringSubstr(json, valueStart + 1, valueEnd - valueStart - 1);
}

double ExtractDoubleValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return 0;
    int colonPos = StringFind(json, ":", keyPos);
    string remaining = StringSubstr(json, colonPos + 1);
    int endPos = StringFind(remaining, ",");
    if(endPos < 0) endPos = StringFind(remaining, "}");
    return StringToDouble(remaining);
}
