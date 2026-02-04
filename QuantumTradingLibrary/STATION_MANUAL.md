# 📟 THE QUANTUM STATION MANUAL
## "The Unhandcuffing Guide"

If I (the AI Assistant) am not here, this is how you stay in control of the DeepCompress Pro system.

### 1. The Master Key
Everything starts with one command. If the computer restarts or you get lost, open a terminal and type:
```powershell
python command_center.py
```

### 2. The Core Components
*   **The Brain (`Signal Bridge`):** This must be running for MT5 to take trades. It calculates the evolution on your CPU/GPU.
*   **The Hands (`ETARE_Executor`):** This is the EA on your MT5 chart. It waits for the Brain to send a `.json` file.
*   **The Archivist (`Archiver Service`):** This captures the market data every 15 minutes. Without this, you can't train new "children" on fresh data.

### 3. Training a "Quantum Child"
When the market regime changes (e.g., from Trending to Choppy), you need a new child. 
1. Open the **Command Center**.
2. Select **[1] TRAIN NEW EXPERT**.
3. It will look at `04_Data/Archive/` and train a new model.
4. The best weights are saved to `etare_redux.db`.

### 4. Troubleshooting
*   **MT5 isn't trading?** Check if `etare_gen_err.log` has errors. Usually, it's just a connection timeout.
*   **Dashboard is blank?** Make sure the `Signal Bridge` is running; it hosts the data the dashboard needs.
*   **Python errors?** Most are solved by restarting the `Command Center`.

### 5. Critical Paths
*   **Signals:** `C:\Users\jjj10\AppData\Roaming\MetaQuotes\Terminal\...\MQL5\Files\etare_signals.json`
*   **Archives:** `04_Data/Archive/`
*   **Database:** `etare_redux.db`

---
**YOU ARE THE ARCHITECT. THE SYSTEM SERVES THE NARRATIVE.**
