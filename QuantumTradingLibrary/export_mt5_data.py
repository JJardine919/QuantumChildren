  # Export BTCUSD M5 data for the last 5 years (60 months)
    btc_symbol = "BTCUSD"
    btc_output_dir = "mt5_historical_data"
    btc_filename = f"{btc_output_dir}/{btc_symbol}_M5_5years.csv"

    os.makedirs(btc_output_dir, exist_ok=True)
    btc_end_date = datetime.now()
    btc_start_date = btc_end_date - timedelta(days=5 * 365)
    rates = mt5.copy_rates_range(btc_symbol, mt5.TIMEFRAME_M5, btc_start_date, btc_end_date)
    if rates is not None and len(rates) > 0:
        btc_df = pd.DataFrame(rates)
        btc_df['time'] = pd.to_datetime(btc_df['time'], unit='s')
        btc_df.to_csv(btc_filename, index=False)
        print(f"✅ BTCUSD: {len(btc_df):,} bars saved to {btc_filename}")
    else:
        print("❌ No BTCUSD data found for the last 5 years.")