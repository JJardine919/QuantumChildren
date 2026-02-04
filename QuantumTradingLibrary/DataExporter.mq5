//+------------------------------------------------------------------+
//|                                              DataExporter.mq5    |
//|                                  Copyright 2026, Quantum Library |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property service

#include <Files\FileTxt.mqh>

//+------------------------------------------------------------------+
//| Service program start function                                   |
//+------------------------------------------------------------------+
void OnStart()
  {
   Print("Starting Data Exporter Service...");
   
   while(!IsStopped())
     {
      ExportData("BTCUSD");
      Sleep(60000); // Wait 1 minute
     }
  }
//+------------------------------------------------------------------+

void ExportData(string symbol)
  {
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(symbol, PERIOD_M5, 0, 1000, rates);
   if(copied <= 0) 
     {
      Print("Failed to copy rates for ", symbol);
      return;
     }
     
   // Manually construct JSON string
   string json = "{\"" + symbol + "\": [";
   
   for(int i=copied-1; i>=0; i--)
     {
      string bar = StringFormat("{\"time\": %I64d, \"open\": %G, \"high\": %G, \"low\": %G, \"close\": %G, \"tick_volume\": %I64d}", 
                                rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
      
      json += bar;
      if(i > 0) json += ",";
     }
   json += "]}";
   
   // Write to file
   int file_handle = FileOpen("market_data.json", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(file_handle != INVALID_HANDLE)
     {
      FileWriteString(file_handle, json);
      FileClose(file_handle);
      Print("Exported ", copied, " bars to market_data.json");
     }
   else
     {
      Print("Failed to open file for writing. Error: ", GetLastError());
     }
  }
