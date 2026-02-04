ya import MetaTrader5 as mt5                                                                                                                                      

  import pandas as pd                                                                                                                                            

  from datetime import datetime, timedelta                                                                                                                       

  import os                                                                                                                                                      

                                                                                                                                                                 

  def export_mt5_data():                                                                                                                                         

      if not mt5.initialize():                                                                                                                                   

          print("MT5 initialization failed")                                                                                                                     

          return                                                                                                                                                 

      os.makedirs("mt5_historical_data", exist_ok=True)                                                                                                          

      end_date = datetime.now()                                                                                                                                  

      start_date = end_date - timedelta(days=1800)                                                                                                               

      print("Exporting BTCUSD...")                                                                                                                               

      rates = mt5.copy_rates_range("BTCUSD", mt5.TIMEFRAME_M5, start_date, end_date)                                                                             

      if rates is None or len(rates) == 0:                                                                                                                       

          print("No data found")                                                                                                                                 

          mt5.shutdown()                                                                                                                                         

          return                                                                                                                                                 

      df = pd.DataFrame(rates)                                                                                                                                   

      df['time'] = pd.to_datetime(df['time'], unit='s')                                                                                                          

      df.to_csv("mt5_historical_data/BTCUSD_M5.csv", index=False)                                                                                                

      print(f"Done - {len(df)} bars saved")                                                                                                                      

      mt5.shutdown()                                                                                                                                             

                                                                                                                                                                 

  if __name__ == "__main__":                                                                                                                                     

      export_mt5_data()                                                                                                                   