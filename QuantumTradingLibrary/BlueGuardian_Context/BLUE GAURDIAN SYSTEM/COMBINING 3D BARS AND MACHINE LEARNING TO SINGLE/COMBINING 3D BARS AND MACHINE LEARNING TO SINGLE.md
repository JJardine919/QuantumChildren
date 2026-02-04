

# COMBINING 3D BARS AND MACHINE LEARNING TO SINGLE







In previous articles, we explored the use of quantum computing to extract nonlinear correlations from market data, as well as the integration of language models with CatBoost gradient boosting. Forecast accuracy was 62.4% in cross-validation, resulting in a return of +27.39% over a month of backtesting on a $140 micro account.



However, analysis revealed that the system misses critical information—the multidimensional structure of the interaction between price, time, and volume. Classic indicators rely on market projections on two-dimensional charts, missing the three-dimensional picture of what's happening. This article presents the full integration of a 3D bar module into a quantum-enhanced trading system.







Integrated system architecture

The system consists of four interconnected modules. The first module receives data from MetaTrader 5 on eight currency pairs on the M15 timeframe. This data is simultaneously fed to three parallel processors: a 3D bar module, a Qiskit-based quantum encoder, and a technical indicator calculation unit.



\# System configuration 

MODEL\_NAME = "koshtenco/quantum-trader-fusion-3d" 

BASE\_MODEL = "llama3.2:3b" 

SYMBOLS = \[ "EURUSD" , "GBPUSD" , "USDCHF" , "USDCAD" , 

&nbsp;           "AUDUSD" , "NZDUSD" , "EURGBP" , "AUDCHF" ]

TIMEFRAME = mt5.TIMEFRAME\_M15

LOOKBACK = 400



\# Quantum parameters 

N\_QUBITS = 8 

N\_SHOTS = 2048



\# 3D Bars parameters 

MIN\_SPREAD\_MULTIPLIER = 45 

VOLUME\_BRICK = 500 

USE\_3D\_BARS = True

The 3D Bars module transforms non-stationary OHLCV data into stationary 4D features. The quantum encoder uses 8 qubits to create 256 quantum states and extract nonlinear correlations. The technical indicators module calculates 33 classic indicators, including RSI, MACD, ATR, and others.



All features are combined and fed into the CatBoost model, which processes 52+ features simultaneously. Gradient boosting is trained to predict the price movement direction in 24 hours. CatBoost's output is optionally processed by the Llama 3.2 3B language model, which adds contextual interpretation to the forecasts.







Bars3D Class: Solving the Problem of Non-Stationarity

The main problem with working with financial time series is their non-stationarity. EURUSD is trading at 1.0850 today, 1.0920 tomorrow, and 1.1500 a year from now. Absolute price values ​​are useless for machine learning, since the model is trained on specific numbers that will never be repeated in the future.



class Bars3D:

&nbsp;    """

&nbsp;   A class for creating stationary 4D features (3D bars)

&nbsp;   Implementation from the article on multidimensional bars

&nbsp;   """

&nbsp;   

&nbsp;   def \_\_init\_\_(self, min\_spread\_multiplier: int = 45 , volume\_brick: int = 500 ):

&nbsp;       self.min\_spread\_multiplier = min\_spread\_multiplier

&nbsp;       self.volume\_brick = volume\_brick

&nbsp;       self.scaler = MinMaxScaler(feature\_range=( 3 , 9 ))

The normalization to the range from 3 to 9 is not random and is related to the harmonics of the numbers 3, 6, and 9, which were used in Gann's theory and Nikola Tesla's research. Empirically, this range produces more stationary series compared to the standard normalization to the interval from zero to one.



The create\_3d\_features method takes a dataframe with OHLCV data and returns an enriched dataframe with stationary features:



def create\_3d\_features(self, df: pd.DataFrame, symbol\_info= None ) -> pd.DataFrame:

&nbsp;    """Creates stationary 4D features from regular OHLCV data""" 

&nbsp;   if  len (df) < 21 :

&nbsp;       log.warning( "Not enough data for 3D bars" )

&nbsp;        return df

&nbsp;   

&nbsp;   d = df.copy()

&nbsp;   

&nbsp;   # Time dimension (cyclic features) 

&nbsp;   if  isinstance (d.index, pd.DatetimeIndex):

&nbsp;       d\[ 'time\_sin' ] = np.sin( 2 \* np.pi \* d.index.hour / 24 )

&nbsp;       d\[ 'time\_cos' ] = np.cos( 2 \* np.pi \* d.index.hour / 24 )

&nbsp;   

&nbsp;   # Price dimension (returns and acceleration) 

&nbsp;   d\[ 'typical\_price' ] = (d\[ 'high' ] + d\[ 'low' ] + d\[ 'close' ]) / 3 

&nbsp;   d\[ 'price\_return' ] = d\[ 'typical\_price' ].pct\_change()

&nbsp;   d\[ 'price\_acceleration' ] = d\[ 'price\_return' ].diff()

&nbsp;   

&nbsp;   # Volume measurement 

&nbsp;   d\[ 'volume\_change' ] = d\[ 'tick\_volume' ].pct\_change()

&nbsp;   d\[ 'volume\_acceleration' ] = d\[ 'volume\_change' ].diff()

&nbsp;   

&nbsp;   # Measuring volatility 

&nbsp;   d\[ 'volatility' ] = d\[ 'price\_return' ].rolling( 20 ).std()

&nbsp;   d\[ 'volatility\_change' ] = d\[ 'volatility' ].pct\_change()

The first dimension represents the temporal structure through cyclical features. The hour of the day is encoded not by a linear number from 0 to 23, but by two functions: sine and cosine. The sine of the hour is calculated as sin(2π × hour / 24), and the cosine is similar. This representation makes 11:00 PM and 12:00 AM mathematically close, unlike the naive approach where 23 and 0 are maximally distant.



The second dimension describes price movement through returns and acceleration. Return is the percentage change in the typical price between bars. Acceleration is the difference between the current and previous returns, i.e., the second derivative of price with respect to time.



The third dimension deals with volume information. Volume change is calculated as the percentage increase in tick volume between bars. Volume acceleration is the difference between the current and previous volume change.



The fourth dimension captures volatility. Volatility is calculated as the standard deviation of returns over a 20-bar rolling window. Volatility change is the percentage increase in volatility between bars.



For each bar starting from the twentieth, a sliding window of 21 points is created:



\# Create normalized features in a sliding window

&nbsp;   bar3d\_features = \[]

&nbsp;   

&nbsp;   for idx in  range ( 20 , len (d)):

&nbsp;       window = d.iloc\[idx- 20 :idx+ 1 ]

&nbsp;       

&nbsp;       features = {

&nbsp;           'bar3d\_price\_return' : float (window\[ 'price\_return' ].iloc\[- 1 ]),

&nbsp;            'bar3d\_price\_accel' : float (window\[ 'price\_acceleration' ].iloc\[- 1 ]),

&nbsp;            'bar3d\_volume\_change' : float (window\[ 'volume\_change' ].iloc\[- 1 ]),

&nbsp;            'bar3d\_volatility\_change' : float (window\[ 'volatility\_change' ].iloc\[- 1 ]),

&nbsp;            'bar3d\_volume\_accel' : float (window\[ 'volume\_acceleration' ].iloc\[- 1 ]),

&nbsp;            'bar3d\_time\_sin' : float (d.iloc\[idx]\[ 'time\_sin' ]),

&nbsp;            'bar3d\_time\_cos' : float (d.iloc\[idx]\[ 'time\_cos' ]),

&nbsp;            'bar3d\_price\_velocity' : float (window\[ 'price\_acceleration' ].mean()),

&nbsp;            'bar3d\_volume\_intensity' : float (window\[ 'volume\_change' ].mean()),

&nbsp;            'bar3d\_price\_change\_mean' : float (window\[ 'price\_return' ].mean()),

&nbsp;       }

&nbsp;       

&nbsp;       bar3d\_features.append(features)

All features are combined into a dataframe and normalized using the MinMaxScaler scaler to a range from 3 to 9. Normalization is applied only to non-zero rows; missing values ​​are filled using backpropagation:

\# Normalize to range 3-9 

&nbsp;   cols\_to\_scale = \[col for col in bar3d\_df.columns if col.startswith( 'bar3d\_' )]

&nbsp;    if cols\_to\_scale:

&nbsp;       result\[cols\_to\_scale] = result\[cols\_to\_scale].bfill().fillna( 0 )

&nbsp;       

&nbsp;       mask = result\[cols\_to\_scale]. abs (). sum (axis= 1 ) > 0 

&nbsp;       if mask. sum () > 0 :

&nbsp;           result.loc\[mask, cols\_to\_scale] = self.scaler.fit\_transform(

&nbsp;               result.loc\[mask, cols\_to\_scale]

&nbsp;           )

An analysis of over 400,000 EURUSD bars for the period 2022-2024 revealed an interesting pattern. When the 70% quantiles of price and volume volatility are simultaneously exceeded, there is an increased likelihood of a price reversal in the next few bars.



\# Additional metrics 

result\[ 'bar3d\_price\_volatility' ] = result\[ 'bar3d\_price\_change\_mean' ].rolling( 10 ).std()

result\[ 'bar3d\_volume\_volatility' ] = result\[ 'bar3d\_volume\_change' ].rolling( 10 ).std()



\# Yellow cluster detector (reversal predictor) 

result\[ 'bar3d\_yellow\_cluster' ] = (

&nbsp;   (result\[ 'bar3d\_price\_volatility' ] > result\[ 'bar3d\_price\_volatility' ].quantile( 0.7 )) \&

&nbsp;   (result\[ 'bar3d\_volume\_volatility' ] > result\[ 'bar3d\_volume\_volatility' ].quantile( 0.7 ))

).astype( float )



\# Reversal probability based on yellow clusters 

result\[ 'bar3d\_reversal\_prob' ] = result\[ 'bar3d\_yellow\_cluster' ].rolling( 7 , center= True ).mean()

The detector is implemented using a logical condition. The 70% quantile of second-order price volatility is calculated over the entire available time series. The 70% quantile of volume volatility is calculated similarly. For each bar, it is checked whether the current price volatility exceeds its quantile and whether the current volume volatility exceeds its quantile.



The reversal probability is calculated as the moving average of the yellow cluster in a 7-bar window with centering. Centering means the window looks three bars back, the current bar, and three bars ahead. This yields the local density of yellow clusters around the current point.



The physical meaning of the yellow cluster is as follows: price moves with abnormally high volatility relative to its historical distribution, while volume simultaneously demonstrates instability in order flow. The combination of these two factors creates a state of maximum uncertainty. Most traders are in positions that turn out to be wrong. Smart money begins to reverse positions against the crowd.



The direction of movement and the strength of the trend are calculated as follows:



\# Trend direction 

result\[ 'bar3d\_direction' ] = np.sign(result\[ 'bar3d\_price\_return' ])



\# Trend Counter

trend\_count = \[]

count = 1 

prev\_dir = 0



for direction in result\[ 'bar3d\_direction' ]:

&nbsp;    if pd.isna(direction):

&nbsp;       trend\_count.append( 0 )

&nbsp;        continue

&nbsp;   

&nbsp;   if direction == prev\_dir:

&nbsp;       count += 1 

&nbsp;   else :

&nbsp;       count = 1

&nbsp;   

&nbsp;   trend\_count.append(count)

&nbsp;   prev\_dir = direction



result\[ 'bar3d\_trend\_count' ] = trend\_count

result\[ 'bar3d\_trend\_strength' ] = result\[ 'bar3d\_trend\_count' ] \* result\[ 'bar3d\_direction' ]



Quantum encoder based on Qiskit

The QuantumEncoder class implements quantum feature encoding. The constructor accepts the number of qubits and the number of dimensions. Eight qubits create a space of 2^8, or 256 possible basis states:



class QuantumEncoder:

&nbsp;    """Quantum encoder based on Qiskit"""

&nbsp;   

&nbsp;   def \_\_init\_\_(self, n\_qubits: int = 8 , n\_shots: int = 2048 ):

&nbsp;       self.n\_qubits = n\_qubits

&nbsp;       self.n\_shots = n\_shots

&nbsp;       self.simulator = AerSimulator()

The encode\_and\_measure method accepts an array of features and returns a dictionary with four quantum metrics. The first step is to normalize the features into rotation angles:



def encode\_and\_measure(self, features: np.ndarray) -> Dict \[ str , float ]:

&nbsp;    """Encodes features into a quantum circuit"""

&nbsp;   

&nbsp;   # Normalize to angles \[0, π] 

&nbsp;   normalized = (features - features. min ()) / (features. max () - features. min () + 1e-8 )

&nbsp;   angles = normalized \* np.pi

&nbsp;   

&nbsp;   # Creating a quantum circuit

&nbsp;   qc = QuantumCircuit(self.n\_qubits, self.n\_qubits)

&nbsp;   

&nbsp;   # RY rotations for feature encoding 

&nbsp;   for i in  range ( min ( len (angles), self.n\_qubits)):

&nbsp;       qc.ry(angles\[i], i)

&nbsp;   

&nbsp;   # Creating entanglement via CZ gates (ring topology) 

&nbsp;   for i in  range (self.n\_qubits - 1 ):

&nbsp;       qc.cz(i, i + 1 )

&nbsp;   qc.cz(self.n\_qubits - 1 , 0 )   # Closing the ring

&nbsp;   

&nbsp;   # Measurement 

&nbsp;   qc.measure( range (self.n\_qubits), range (self.n\_qubits))

For each qubit, an RY rotation is applied by the corresponding angle. The RY gate rotates the qubit around the Y axis of the Bloch sphere by a given angle. Mathematically, this transforms the qubit from the ground state |0⟩ to the superposition cos(θ/2)|0⟩ + sin(θ/2)|1⟩.



The third step creates quantum entanglement between the qubits. A Controlled-Z gate is applied between each pair of adjacent qubits. Successive applications of a Controlled-Z gate between the qubits create a chain of correlations. Additionally, a Controlled-Z gate is applied between the last qubit (7) and the first qubit (0), closing the chain into a ring.



The scheme is run on the simulator 2048 times. A probability array of length 256 is generated from the calculation dictionary:



\# Run the simulation

&nbsp;   job = self.simulator.run(qc, shots=self.n\_shots)

&nbsp;   result = job.result()

&nbsp;   counts = result.get\_counts()

&nbsp;   

&nbsp;   # Convert to an array of probabilities 

&nbsp;   total\_shots = sum (counts.values())

&nbsp;   probabilities = np.array(\[

&nbsp;       counts.get( format (i, f'0 {self.n\_qubits} b' ), 0 ) / total\_shots 

&nbsp;        for i in  range ( 2 \*\*self.n\_qubits)

&nbsp;   ])

&nbsp;   

&nbsp;   # Extracting quantum features 

&nbsp;   quantum\_entropy = entropy(probabilities + 1e-10 , base= 2 )

&nbsp;   dominant\_state\_prob = np. max (probabilities)

&nbsp;   significant\_states = np. sum (probabilities > 0.03 )

&nbsp;   quantum\_variance = np.var(probabilities)

&nbsp;   

&nbsp;   return {

&nbsp;        'quantum\_entropy' : quantum\_entropy,

&nbsp;        'dominant\_state\_prob' : dominant\_state\_prob,

&nbsp;        'significant\_states' : significant\_states,

&nbsp;        'quantum\_variance' : quantum\_variance

&nbsp;   }

Four quantum features are extracted from the probability distribution. Quantum entropy, using Shannon's formula, is calculated as the sum over all states of the product of the probability and the logarithm of the probability to base two. Entropy is measured in bits and ranges from zero to eight.



High entropy—above 6.5—indicates a market in a state of uncertainty; low entropy—below 4.5—indicates a settled market. The probability of a dominant state is calculated as the maximum value among all 256 probabilities. The number of significant states calculates how many states have a probability above the 3% threshold. Quantum variance is the standard variance of a probability array.





Technical Indicators: 33 Classic Signs

The calculate\_features function takes a dataframe with OHLCV data, an optional Bars3D instance, and symbol information:



def calculate\_features(df: pd.DataFrame, bars\_3d: Bars3D = None , symbol\_info= None ) -> pd.DataFrame:

&nbsp;    """Calculation of technical indicators + 3D bars"""

&nbsp;   d = df.copy()

&nbsp;   d\[ "close\_prev" ] = d\[ "close" ].shift( 1 )

&nbsp;   

&nbsp;   # ATR

&nbsp;   tr = pd.concat(\[

&nbsp;       d\[ "high" ] - d\[ "low" ],

&nbsp;       (d\[ "high" ] - d\[ "close\_prev" ]). abs (),

&nbsp;       (d\[ "low" ] - d\[ "close\_prev" ]). abs (),

&nbsp;   ], axis= 1 ). max (axis= 1 )

&nbsp;   d\[ "ATR" ] = tr.rolling( 14 ).mean()

&nbsp;   

&nbsp;   # RSI 

&nbsp;   delta = d\[ "close" ].diff()

&nbsp;   up = delta.clip(lower= 0 ).rolling( 14 ).mean()

&nbsp;   down = (-delta.clip(upper= 0 )).rolling( 14 ).mean()

&nbsp;   rs = up / down.replace( 0 , np.nan)

&nbsp;   d\[ "RSI" ] = 100 - ( 100 / ( 1 + rs))

&nbsp;   

&nbsp;   # MACD 

&nbsp;   ema12 = d\[ "close" ].ewm(span= 12 , adjust= False ).mean()

&nbsp;   ema26 = d\[ "close" ].ewm(span= 26 , adjust= False ).mean()

&nbsp;   d\[ "MACD" ] = ema12 - ema26

&nbsp;   d\[ "MACD\_signal" ] = d\[ "MACD" ].ewm(span= 9 , adjust= False ).mean()

&nbsp;   

&nbsp;   # Bollinger Bands 

&nbsp;   d\[ "BB\_middle" ] = d\[ "close" ].rolling( 20 ).mean()

&nbsp;   bb\_std = d\[ "close" ].rolling( 20 ).std()

&nbsp;   d\[ "BB\_upper" ] = d\[ "BB\_middle" ] + 2 \* bb\_std

&nbsp;   d\[ "BB\_lower" ] = d\[ "BB\_middle" ] - 2 \* bb\_std

&nbsp;   d\[ "BB\_position" ] = (d\[ "close" ] - d\[ "BB\_lower" ]) / (d\[ "BB\_upper" ] - d\[ "BB\_lower" ])

&nbsp;   

&nbsp;   # Stochastic 

&nbsp;   low\_14 = d\[ "low" ].rolling( 14 ). min ()

&nbsp;   high\_14 = d\[ "high" ].rolling( 14 ). max ()

&nbsp;   d\[ "Stoch\_K" ] = 100 \* (d\[ "close" ] - low\_14) / (high\_14 - low\_14)

&nbsp;   d\[ "Stoch\_D" ] = d\[ "Stoch\_K" ].rolling( 3 ).mean()

&nbsp;   

&nbsp;   # EMA 

&nbsp;   d\[ "EMA\_50" ] = d\[ "close" ].ewm(span= 50 , adjust= False ).mean()

&nbsp;   d\[ "EMA\_200" ] = d\[ "close" ].ewm(span= 200 , adjust= False ).mean()

&nbsp;   

&nbsp;   # Volumes and yields 

&nbsp;   d\[ "vol\_ratio" ] = d\[ "tick\_volume" ] / d\[ "tick\_volume" ].rolling( 20 ).mean()

&nbsp;   d\[ "price\_change\_1" ] = d\[ "close" ].pct\_change( 1 )

&nbsp;   d\[ "price\_change\_5" ] = d\[ "close" ].pct\_change( 5 )

&nbsp;   d\[ "price\_change\_21" ] = d\[ "close" ].pct\_change( 21 )

&nbsp;   d\[ "volatility\_20" ] = d\[ "price\_change\_1" ].rolling( 20 ).std()

&nbsp;   

&nbsp;   # 3D Bars Integration 

&nbsp;   if USE\_3D\_BARS and bars\_3d is  not  None :

&nbsp;       d = bars\_3d.create\_3d\_features(d, symbol\_info)

&nbsp;   

&nbsp;   return d.dropna()

Average True Range is calculated as a moving average of the true range over a 14-bar window. True range captures volatility, taking into account gaps between bars. RSI is calculated as 100 minus 100 divided by one plus RS, where RS is the ratio of average advance to average decline.



MACD is the difference between two exponential moving averages with periods of 12 and 26 bars. Bollinger Bands are constructed around a moving average of price plus or minus two standard deviations. The Stochastic Oscillator is calculated using the minimum and maximum values ​​over 14 bars.





Training CatBoost on Joint Features

The train\_catboost\_model function takes a dictionary of symbol dataframes, a quantum encoder instance, and a Bars3D instance:



def train\_catboost\_model(data\_dict: Dict \[ str , pd.DataFrame],

&nbsp;                       quantum\_encoder: QuantumEncoder,

&nbsp;                       bars\_3d: Bars3D = None ) -> CatBoostClassifier:

&nbsp;    """Trains CatBoost on data with quantum features + 3D bars"""

&nbsp;   

&nbsp;   all\_features = \[]

&nbsp;   all\_targets = \[]

&nbsp;   

&nbsp;   for symbol, df in data\_dict.items():

&nbsp;       symbol\_info = mt5.symbol\_info(symbol)

&nbsp;       df\_features = calculate\_features(df, bars\_3d, symbol\_info)

&nbsp;       

&nbsp;       for idx in  range (LOOKBACK, len (df\_features) - PREDICTION\_HORIZON):

&nbsp;           row = df\_features.iloc\[idx]

&nbsp;           future\_row = df\_features.iloc\[idx + PREDICTION\_HORIZON]

&nbsp;           

&nbsp;           # Target variable: UP (1) if the price in 24 hours is above 

&nbsp;           target = 1  if future\_row\[ 'close' ] > row\[ 'close' ] else  0

&nbsp;           

&nbsp;           # Quantum coding

&nbsp;           feature\_vector = np.array(\[

&nbsp;               row\[ 'RSI' ], row\[ 'MACD' ], row\[ 'ATR' ], row\[ 'vol\_ratio' ],

&nbsp;               row\[ 'BB\_position' ], row\[ 'Stoch\_K' ],

&nbsp;               row\[ 'price\_change\_1' ], row\[ 'volatility\_20' ]

&nbsp;           ])

&nbsp;           quantum\_feats = quantum\_encoder.encode\_and\_measure(feature\_vector)

&nbsp;           

&nbsp;           # Combining all features

&nbsp;           features = {

&nbsp;               'RSI' : row\[ 'RSI' ], 'MACD' : row\[ 'MACD' ], 'ATR' : row\[ 'ATR' ],

&nbsp;                'vol\_ratio' : row\[ 'vol\_ratio' ], 'BB\_position' : row\[ 'BB\_position' ],

&nbsp;                'Stoch\_K' : row\[ 'Stoch\_K' ], 'Stoch\_D' : row\[ 'Stoch\_D' ],

&nbsp;                'EMA\_50' : row\[ 'EMA\_50' ], 'EMA\_200' : row\[ 'EMA\_200' ],

&nbsp;                'price\_change\_1' : row\[ 'price\_change\_1' ],

&nbsp;                'price\_change\_5' : row\[ 'price\_change\_5' ],

&nbsp;                'price\_change\_21' : row\[ 'price\_change\_21' ],

&nbsp;                'volatility\_20' : row\[ 'volatility\_20' ],

&nbsp;                'quantum\_entropy' : quantum\_feats\[ 'quantum\_entropy' ],

&nbsp;                'dominant\_state\_prob' : quantum\_feats\[ 'dominant\_state\_prob' ],

&nbsp;                'significant\_states' : quantum\_feats\[ 'significant\_states' ],

&nbsp;                'quantum\_variance' : quantum\_feats\[ 'quantum\_variance' ],

&nbsp;                'symbol' : symbol

&nbsp;           }

&nbsp;           

&nbsp;           # Add 3D bars if available 

&nbsp;           if USE\_3D\_BARS and  'bar3d\_price\_return'  in row:

&nbsp;               features.update({

&nbsp;                   'bar3d\_yellow\_cluster' : row.get( 'bar3d\_yellow\_cluster' , 0 ),

&nbsp;                    'bar3d\_reversal\_prob' : row.get( 'bar3d\_reversal\_prob' , 0 ),

&nbsp;                    'bar3d\_trend\_strength' : row.get( 'bar3d\_trend\_strength' , 0 ),

&nbsp;                    'bar3d\_price\_volatility' : row.get( 'bar3d\_price\_volatility' , 0 ),

&nbsp;                    'bar3d\_volume\_volatility' : row.get( 'bar3d\_volume\_volatility' , 0 ),

&nbsp;               })

&nbsp;           

&nbsp;           all\_features.append(features)

&nbsp;           all\_targets.append(target)

For each position, the current bar and future bar are extracted using PREDICTION\_HORIZON positions ahead. The target variable is set to one if the future bar's closing price is higher than the current one, and zero otherwise. A feature vector for quantum coding is created from eight key technical indicators.



A dictionary of features is generated for the current bar. All 33 technical indicators are included, four quantum features are added, and the symbol name is added as a categorical feature. If the USE\_3D\_BARS flag is enabled, five key features from 3D bars are added.



Training a model with TimeSeriesSplit cross-validation:



X = pd.DataFrame(all\_features)

&nbsp;   y = np.array(all\_targets)

&nbsp;   X = pd.get\_dummies(X, columns=\[ 'symbol' ], prefix= 'sym' )

&nbsp;   

&nbsp;   model = CatBoostClassifier(

&nbsp;       iterations= 3000 ,

&nbsp;       learning\_rate= 0.03 ,

&nbsp;       depth= ​​8 ,

&nbsp;       loss\_function= 'Logloss' ,

&nbsp;       eval\_metric= 'Accuracy' ,

&nbsp;       random\_seed= 42 ,

&nbsp;       verbose= 500

&nbsp;   )

&nbsp;   

&nbsp;   from sklearn.model\_selection import TimeSeriesSplit

&nbsp;   tscv = TimeSeriesSplit(n\_splits= 3 )

&nbsp;   

&nbsp;   accuracies = \[]

&nbsp;   for fold\_idx, (train\_idx, val\_idx) in  enumerate (tscv.split(X)):

&nbsp;       X\_train, X\_val = X.iloc\[train\_idx], X.iloc\[val\_idx]

&nbsp;       y\_train, y\_val = y\[train\_idx], y\[val\_idx]

&nbsp;       

&nbsp;       model.fit(X\_train, y\_train, eval\_set=(X\_val, y\_val), verbose= False )

&nbsp;       accuracy = model.score(X\_val, y\_val)

&nbsp;       accuracies.append(accuracy)

&nbsp;       print ( f"Fold {fold\_idx + 1 } Accuracy: {accuracy\* 100 : .2 f} %" )

&nbsp;   

&nbsp;   print ( f"Average accuracy: {np.mean(accuracies)\* 100 : .2 f} % ± {np.std(accuracies)\* 100 : .2 f} %" )

TimeSeriesSplit splits the data sequentially over time. For three folds, the first is trained on the first 33% of the data and validated on the next 33%. The second is trained on the first 67% and validated on the last 33%. The third is trained on all data except the last 33% and validated on them.



Feature importance analysis shows the contribution of 3D bars:



\# Train the final model on all data 

&nbsp;   model.fit(X, y, verbose= 500 )

&nbsp;   model.save\_model( "models/catboost\_quantum\_3d.cbm" )

&nbsp;   

&nbsp;   # Analysis of feature importance

&nbsp;   feature\_importance = model.get\_feature\_importance()

&nbsp;   feature\_names = X.columns

&nbsp;   importance\_df = pd.DataFrame({

&nbsp;       'feature' : feature\_names,

&nbsp;        'importance' : feature\_importance

&nbsp;   }).sort\_values( 'importance' , ascending= False )

&nbsp;   

&nbsp;   print ( "TOP 10 IMPORTANT FEATURES:" )

&nbsp;    print (importance\_df.head( 10 ))

&nbsp;   

&nbsp;   # Check 3D bars in the top 

&nbsp;   if USE\_3D\_BARS:

&nbsp;       bar3d\_features = importance\_df\[importance\_df\[ 'feature' ]. str .startswith( 'bar3d\_' )]

&nbsp;        print ( f"\\nTOP 3D BARS ( {len(bar3d\_features)} features):" )

&nbsp;        print (bar3d\_features.head( 10 ))

The training results show an average accuracy of 65.8% with a standard deviation of 0.5%. This is 3.4 percentage points higher than the previous version without 3D bars. The top 10 important features include bar3d\_yellow\_cluster in first place with an importance of 18.7%, quantum\_entropy in second place with 16.2%, and bar3d\_reversal\_prob in third place with 12.4%.





Backtesting: from $140 to $193

The backtest function backtests the trained model on historical data for the last 30 days:



def backtest(catboost\_model, use\_llm= False ):

&nbsp;    """Backtesting with CatBoost + Quantum + 3D"""

&nbsp;   

&nbsp;   end = datetime.now().replace(second= 0 , microsecond= 0 )

&nbsp;   start = end - timedelta(days=BACKTEST\_DAYS)

&nbsp;   

&nbsp;   # Loading data

&nbsp;   data = {}

&nbsp;   for sym in SYMBOLS:

&nbsp;       rates = mt5.copy\_rates\_range(sym, TIMEFRAME, start, end)

&nbsp;       if rates is  None  or  len (rates) == 0 :

&nbsp;            continue

&nbsp;       df = pd.DataFrame(rates)

&nbsp;       df\[ "time" ] = pd.to\_datetime(df\[ "time" ], unit= "s" )

&nbsp;       df.set\_index( "time" , inplace= True )

&nbsp;        if  len (df) > LOOKBACK + PREDICTION\_HORIZON:

&nbsp;           data\[sym] = df

&nbsp;   

&nbsp;   balance = INITIAL\_BALANCE

&nbsp;   trades = \[]

&nbsp;   

&nbsp;   quantum\_encoder = QuantumEncoder(N\_QUBITS, N\_SHOTS)

&nbsp;   bars\_3d = Bars3D(MIN\_SPREAD\_MULTIPLIER, VOLUME\_BRICK)

&nbsp;   

&nbsp;   # Analysis points every 24 hours 

&nbsp;   main\_symbol = list (data.keys())\[ 0 ]

&nbsp;   main\_data = data\[main\_symbol]

&nbsp;   total\_bars = len (main\_data)

&nbsp;   analysis\_points = list ( range (LOOKBACK, total\_bars - PREDICTION\_HORIZON, PREDICTION\_HORIZON))

For each analysis point, the system loads historical data up to the current moment, calculates all features including 3D bars, performs quantum encoding, and receives a forecast from CatBoost:

for point\_idx, current\_idx in  enumerate (analysis\_points):

&nbsp;       current\_time = main\_data.index\[current\_idx]

&nbsp;       

&nbsp;       for sym in SYMBOLS:

&nbsp;           historical\_data = data\[sym].iloc\[:current\_idx + 1 ].copy()

&nbsp;           symbol\_info = mt5.symbol\_info(sym)

&nbsp;           

&nbsp;           df\_with\_features = calculate\_features(historical\_data, bars\_3d, symbol\_info)

&nbsp;           row = df\_with\_features.iloc\[- 1 ]

&nbsp;           

&nbsp;           # Quantum coding

&nbsp;           feature\_vector = np.array(\[

&nbsp;               row\[ 'RSI' ], row\[ 'MACD' ], row\[ 'ATR' ], row\[ 'vol\_ratio' ],

&nbsp;               row\[ 'BB\_position' ], row\[ 'Stoch\_K' ],

&nbsp;               row\[ 'price\_change\_1' ], row\[ 'volatility\_20' ]

&nbsp;           ])

&nbsp;           quantum\_feats = quantum\_encoder.encode\_and\_measure(feature\_vector)

&nbsp;           

&nbsp;           # Forming features for CatBoost

&nbsp;           X\_features = {

&nbsp;               'RSI' : row\[ 'RSI' ], 'MACD' : row\[ 'MACD' ], 'ATR' : row\[ 'ATR' ],

&nbsp;                # ... all 33 technical 

&nbsp;               'quantum\_entropy' : quantum\_feats\[ 'quantum\_entropy' ],

&nbsp;                'dominant\_state\_prob' : quantum\_feats\[ 'dominant\_state\_prob' ],

&nbsp;                'significant\_states' : quantum\_feats\[ 'significant\_states' ],

&nbsp;                'quantum\_variance' : quantum\_feats\[ 'quantum\_variance' ],

&nbsp;           }

&nbsp;           

&nbsp;           # Add 3D features 

&nbsp;           if  'bar3d\_yellow\_cluster'  in row:

&nbsp;               X\_features.update({

&nbsp;                   'bar3d\_yellow\_cluster' : row.get( 'bar3d\_yellow\_cluster' , 0 ),

&nbsp;                    'bar3d\_reversal\_prob' : row.get( 'bar3d\_reversal\_prob' , 0 ),

&nbsp;                    'bar3d\_trend\_strength' : row.get( 'bar3d\_trend\_strength' , 0 ),

&nbsp;                    'bar3d\_price\_volatility' : row.get( 'bar3d\_price\_volatility' , 0 ),

&nbsp;                    'bar3d\_volume\_volatility' : row.get( 'bar3d\_volume\_volatility' , 0 ),

&nbsp;               })

&nbsp;           

&nbsp;           # CatBoost Forecast

&nbsp;           X\_df = pd.DataFrame(\[X\_features])

&nbsp;           for s in SYMBOLS:

&nbsp;               X\_df\[ f'sym\_ {s} ' ] = 1  if s == sym else  0

&nbsp;           

&nbsp;           proba = catboost\_model.predict\_proba(X\_df)\[ 0 ]

&nbsp;           catboost\_direction = "UP"  if proba\[ 1 ] > 0.5  else  "DOWN" 

&nbsp;           catboost\_confidence = max (proba) \* 100

&nbsp;           

&nbsp;           # Check for yellow cluster 

&nbsp;           if row.get( 'bar3d\_yellow\_cluster' , 0 ) > 0.5 :

&nbsp;                print ( f" YELLOW CLUSTER!" )

The presence of a yellow cluster is checked and a warning is issued. If the final confidence level is above the minimum threshold, a virtual trade is opened, taking into account all costs:

if final\_confidence < MIN\_PROB:

&nbsp;                continue

&nbsp;           

&nbsp;           # Calculate the result after 24 hours

&nbsp;           exit\_idx = current\_idx + PREDICTION\_HORIZON

&nbsp;           exit\_row = data\[sym].iloc\[exit\_idx]

&nbsp;           

&nbsp;           # Spread accounting 

&nbsp;           entry\_price = row\[ 'close' ] + SPREAD\_PIPS \* point if final\_direction == "UP"  else row\[ 'close' ]

&nbsp;           exit\_price = exit\_row\[ 'close' ] if final\_direction == "UP"  else exit\_row\[ 'close' ] + SPREAD\_PIPS \* point

&nbsp;           

&nbsp;           # Price movement in points 

&nbsp;           price\_move\_pips = (exit\_price - entry\_price) / point if final\_direction == "UP"  else (entry\_price - exit\_price) / point

&nbsp;           

&nbsp;           # Position sizing based on ATR

&nbsp;           risk\_amount = balance \* RISK\_PER\_TRADE

&nbsp;           atr\_pips = row\[ 'ATR' ] / point

&nbsp;           stop\_loss\_pips = max ( 20 , atr\_pips \* 2 )

&nbsp;           lot\_size = risk\_amount / (stop\_loss\_pips \* point \* contract\_size)

&nbsp;           lot\_size = max ( 0.01 , min (lot\_size, 10.0 ))

&nbsp;           

&nbsp;           # Profit taking into account swap and slippage

&nbsp;           profit\_usd = price\_move\_pips \* point \* contract\_size \* lot\_size

&nbsp;           profit\_usd -= swap\_cost \* (lot\_size / 0.01 )

&nbsp;           profit\_usd -= SLIPPAGE \* point \* contract\_size \* lot\_size

&nbsp;           

&nbsp;           balance += profit\_usd

After processing all analysis points, final statistics are calculated. These include the total number of trades, the number of profitable trades, win rate, average profit and loss, profit factor, maximum drawdown, and sharpe ratio.









Conclusion

The 3D Bars module increased the trading system's efficiency: accuracy +3.4 percentage points, win rate +3.85%, and profitability +10.66%. This confirms the value of multidimensional market analysis.



The yellow cluster detector identifies areas of increased volatility. With coverage of ~40% of reversals, the signal is highly specific and useful for risk management.



Three of the five key features are derived from 3D bars. The bar3d\_yellow\_cluster feature is the most important (18.7%), ahead of quantum entropy (16.2%).



The system is implemented in a single Python file (1,691 lines). It uses MetaTrader5, Qiskit, CatBoost, Ollama, NumPy, Pandas, and Scikit-learn. Training time: 2–3 hours on CPU. Forecast time: ~3 seconds for 8 currency pairs.



Limitations: 3D bar plotting takes 5–10 minutes for 15,000 candles; parameters are configured for EURUSD M15 and require adaptation for other markets; market instability requires retraining every 1–2 months.



Further developments: multi-timeframe analysis, use of real quantum processors, expansion to other assets, dynamic parameter adjustment, model ensembles.

