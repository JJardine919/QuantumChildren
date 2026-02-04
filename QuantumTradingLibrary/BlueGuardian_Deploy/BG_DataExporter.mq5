//+------------------------------------------------------------------+
//|                                           BG_DataExporter.mq5    |
//|                                  Copyright 2026, Quantum Library |
//|                     Blue Guardian Data Exporter - Multi-Symbol   |
//|                               v2.1 - Production Ready            |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "2.10"
#property service

//--- Configuration
input string Symbols = "BTCUSD";           // Symbols to export (comma-separated)
input int    BarsToExport = 500;           // Number of bars to export
input int    UpdateInterval = 30;          // Update interval (seconds)
input bool   UseCommonFolder = true;       // Use common folder (for VPS)

//+------------------------------------------------------------------+
//| Service program start function                                   |
//+------------------------------------------------------------------+
void OnStart()
{
    Print("========================================");
    Print("BG DATA EXPORTER v2.1");
    Print("Symbols: ", Symbols);
    Print("Bars: ", BarsToExport);
    Print("Interval: ", UpdateInterval, "s");
    Print("Common Folder: ", UseCommonFolder);
    Print("========================================");

    // Parse symbols
    string symbolList[];
    int count = StringSplit(Symbols, ',', symbolList);

    // Trim whitespace from symbols
    for(int i = 0; i < count; i++)
    {
        StringTrimLeft(symbolList[i]);
        StringTrimRight(symbolList[i]);
    }

    while(!IsStopped())
    {
        string json = "{";
        bool firstSymbol = true;
        int exportedCount = 0;

        for(int i = 0; i < count; i++)
        {
            string symbol = symbolList[i];
            if(StringLen(symbol) == 0) continue;

            string symbolData = ExportSymbol(symbol);
            if(StringLen(symbolData) > 0)
            {
                if(!firstSymbol) json += ",";
                json += "\"" + symbol + "\":" + symbolData;
                firstSymbol = false;
                exportedCount++;
            }
        }

        // Add metadata with timestamp
        if(!firstSymbol) json += ",";
        json += "\"_meta\":{";
        json += "\"exported_at\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";
        json += "\"bar_count\":" + IntegerToString(BarsToExport) + ",";
        json += "\"symbols_exported\":" + IntegerToString(exportedCount);
        json += "}";

        json += "}";

        // Write to file
        int flags = FILE_WRITE|FILE_TXT|FILE_ANSI;
        if(UseCommonFolder) flags |= FILE_COMMON;

        int fileHandle = FileOpen("market_data.json", flags);
        if(fileHandle != INVALID_HANDLE)
        {
            FileWriteString(fileHandle, json);
            FileClose(fileHandle);

            MqlDateTime now;
            TimeCurrent(now);
            Print(StringFormat("[%02d:%02d:%02d] Exported %d symbols to market_data.json",
                             now.hour, now.min, now.sec, exportedCount));
        }
        else
        {
            Print("ERROR: Could not write market_data.json - ", GetLastError());
        }

        Sleep(UpdateInterval * 1000);
    }

    Print("Data Exporter stopped");
}

//+------------------------------------------------------------------+
//| Export single symbol data                                        |
//+------------------------------------------------------------------+
string ExportSymbol(string symbol)
{
    // Ensure symbol is selected
    if(!SymbolSelect(symbol, true))
    {
        Print("WARNING: Could not select symbol ", symbol);
        return "";
    }

    // Get symbol digits for proper precision
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);

    MqlRates rates[];
    ArraySetAsSeries(rates, true);

    int copied = CopyRates(symbol, PERIOD_M5, 0, BarsToExport, rates);
    if(copied <= 0)
    {
        Print("WARNING: Could not copy rates for ", symbol, " - Error: ", GetLastError());
        return "";
    }

    // Build JSON array more efficiently
    string data = "[";

    // Export oldest to newest (reverse order since array is series)
    for(int i = copied - 1; i >= 0; i--)
    {
        if(i < copied - 1) data += ",";

        data += "{\"time\":" + IntegerToString(rates[i].time);
        data += ",\"open\":" + DoubleToString(rates[i].open, digits);
        data += ",\"high\":" + DoubleToString(rates[i].high, digits);
        data += ",\"low\":" + DoubleToString(rates[i].low, digits);
        data += ",\"close\":" + DoubleToString(rates[i].close, digits);
        data += ",\"tick_volume\":" + IntegerToString(rates[i].tick_volume);
        data += "}";
    }

    data += "]";
    return data;
}
//+------------------------------------------------------------------+
