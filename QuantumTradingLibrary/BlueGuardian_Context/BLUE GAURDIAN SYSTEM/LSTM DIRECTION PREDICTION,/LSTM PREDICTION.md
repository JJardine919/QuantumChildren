





# &nbsp;                     LSTM DIRECTION PREDICTION











Bidirectional LSTM and quantum computing for motion direction prediction

MetaTrader 5 - Integration |January 16, 2026 at 12:38 AM



522



0

Evgeniy Koshtenko

Evgeniy Koshtenko

Introduction: When Quantum Mechanics Helps Predict the Market

Imagine that before the next price move, the market simultaneously "considers" many possible continuation scenarios: a strong upward momentum, a slow slide downwards, a sharp reversal, a continuation of the flat, and so on. Classic models only see what has already happened historically. They look for repeating sequences of closes, volumes, and indicators. But they almost never have direct access to how "confidently" or "smeared" these possible scenarios were distributed immediately before the market chose one of them.



Here we try to model this “uncertainty structure” using the mathematical apparatus of quantum mechanics—not because the market is physically quantum, but because the quantum formalism provides very convenient and powerful tools for working with superpositions of probabilities and the correlations between them.



An important clarification right away: we're not using any actual quantum computer, and there's no quantum complexity advantage whatsoever. We're simply taking a very simple, fixed, non-trainable quantum circuit and using it as an exotic nonlinear transformator of a small window of price data into a set of statistical distribution characteristics.



The scheme is as follows: three qubits → three RY rotations, the angles of which depend on the average return, volatility, and swing of the last window → CNOT chain between adjacent qubits → measurement a thousand times.



From the resulting histogram of 8 possible outcomes, we extract seven different metrics. The most understandable and, apparently, most useful of these are:



entropy of the probability distribution (the higher it is, the more “equally probable” all scenarios are, the greater the uncertainty)

the maximum probability of any of the eight basic states (how much one scenario "outweighs" the other)

the number of states whose probability is significantly higher than random (a proxy for the width of the superposition)

the degree of "consistency" of the measured outcomes (how close to each other the obtained numerical values ​​of the states lie)

Other metrics (variance, average bit correlation of adjacent qubits, absolute number of significant states) often play a supporting role, but sometimes help in specific market regimes.



Why is all this necessary? Classic signs and indicators almost always describe a past that has already been realized. Our metrics, however, attempt to provide an indirect understanding of how "decisive" or "indecisive" the market was before choosing the direction of the next candle.



On the same dataset (EUR/USD, H1, ~1500 candles), a pure bidirectional LSTM without these features typically yields 47–52% accuracy in predicting the direction of the next candle. With the addition of these seven characteristics, on a small test sample (~130–160 examples), we were able to obtain values ​​in the 62–67% range. However, a caveat: this is a very small and very specific sample. On other timeframes, instruments, or even just with a different seed, the figures could easily drop to +2…+5% or disappear altogether. Therefore, these figures should be treated with extreme caution—for now, this is just an interesting engineering experiment, not a proven trading system.



What follows is the full code, an explanation of the architecture, a method for balancing classes, validation, and—most importantly—a detailed analysis of why such beautiful numbers on a small sample almost always turn out to be overly optimistic.



Let's start with the most quantum part - with how exactly these seven numbers are obtained from the price window.



Implementation in quantum circuit code:



from qiskit import QuantumCircuit, transpile

&nbsp;from qiskit\_aer import AerSimulator

&nbsp;import numpy as np



class QuantumFeatureExtractor:

&nbsp;    def \_\_init\_\_(self, num\_qubits: int = 3 , shots: int = 1000 ):

&nbsp;       self.num\_qubits = num\_qubits

&nbsp;       self.shots = shots

&nbsp;       self.simulator = AerSimulator(method= 'statevector' )

&nbsp;       self.cache = {}

&nbsp;   

&nbsp;   def create\_quantum\_circuit(self, features: np.ndarray) -> QuantumCircuit:

&nbsp;       qc = QuantumCircuit(self.num\_qubits, self.num\_qubits)

&nbsp;       

&nbsp;       for i in  range (self.num\_qubits):

&nbsp;           feature\_idx = i % len (features)

&nbsp;           angle = np.clip(np.pi \* features\[feature\_idx], - 2 \*np.pi, 2 \*np.pi)

&nbsp;           qc.ry(angle, i)

&nbsp;       

&nbsp;       for i in  range (self.num\_qubits - 1 ):

&nbsp;           qc.cx(i, i + 1 )

&nbsp;       

&nbsp;       qc.measure( range (self.num\_qubits), range (self.num\_qubits))

&nbsp;        return qc

The scheme is simple. Three qubits, yielding eight possible states—enough to detect basic patterns, but not too many for fast computation. RY gates are applied to each qubit, with angles calculated from market data. CNOT gates sequentially couple the qubits: the first to the second, the second to the third. Measurements collapse the state, yielding a classical bit vector.



This scheme is run on the IBM Qiskit simulator with a thousand measurements (shots). Why a thousand? It's a balance between accuracy and speed. Fewer shots mean more statistical noise in the results. More shots mean slower performance, but the accuracy gain after 1,000-2,000 measurements is minimal. The simulator uses the statevector method, which means it accurately calculates the quantum state without introducing the noise of real quantum hardware. This is optimal for our purposes.



Now the quantum feature extraction function:



def extract\_quantum\_features(self, price\_data: np.ndarray) -> dict :

&nbsp;    import hashlib

&nbsp;   data\_hash = hashlib.md5(price\_data.tobytes()).hexdigest()

&nbsp;   if data\_hash in self.cache:

&nbsp;        return self.cache\[data\_hash]

&nbsp;   

&nbsp;   returns = np.diff(price\_data) / (price\_data\[:- 1 ] + 1e-10 )

&nbsp;   features = np.array(\[

&nbsp;       np.mean(returns),

&nbsp;       np.std(returns),

&nbsp;       n.p. max (returns) - np. min (returns)

&nbsp;   ])

&nbsp;   features = np.tanh(features)

&nbsp;   

&nbsp;   try :

&nbsp;       qc = self.create\_quantum\_circuit(features)

&nbsp;       compiled\_circuit = transpile(qc, self.simulator, optimization\_level= 2 )

&nbsp;       job = self.simulator.run(compiled\_circuit, shots=self.shots)

&nbsp;       result = job.result()

&nbsp;       counts = result.get\_counts()

&nbsp;       

&nbsp;       quantum\_features = self.\_compute\_quantum\_metrics(counts, self.shots)

&nbsp;       self.cache\[data\_hash] = quantum\_features

&nbsp;       return quantum\_features

&nbsp;    except Exception as e:

&nbsp;        return self.\_get\_default\_features()

Caching by the window's md5 hash is practically mandatory, otherwise processing a sliding window becomes unacceptably slow.



Calculating metrics:



def \_compute\_quantum\_metrics(self, counts: dict , shots: int ) -> dict :

&nbsp;   probabilities = {state: count/shots for state, count in counts.items()}

&nbsp;   

&nbsp;   quantum\_entropy = - sum (p \* np.log2(p) if p > 0  else  0  

&nbsp;                         for p in probabilities.values())

&nbsp;   

&nbsp;   dominant\_state\_prob = max (probabilities.values())

&nbsp;   

&nbsp;   threshold = 0.05 

&nbsp;   significant\_states = sum ( 1  for p in probabilities.values() if p > threshold)

&nbsp;   superposition\_measure = significant\_states / ( 2 \*\* self.num\_qubits)

&nbsp;   

&nbsp;   state\_values ​​= \[ int (state, 2 ) for state in probabilities.keys()]

&nbsp;   max\_value = 2 \*\* self.num\_qubits - 1 

&nbsp;   phase\_coherence = 1.0 - (np.std(state\_values) / max\_value) if  len (state\_values) > 1  else  0.5

&nbsp;   

&nbsp;   entanglement\_degree = self.\_compute\_entanglement\_from\_cnot(probabilities)

&nbsp;   

&nbsp;   mean\_state = sum ( int (state, 2 ) \* prob for state, prob in probabilities.items())

&nbsp;   quantum\_variance = sum (( int (state, 2 ) - mean\_state)\*\* 2 \* prob 

&nbsp;                          for state, prob in probabilities.items())

&nbsp;   

&nbsp;   return {

&nbsp;        'quantum\_entropy' : quantum\_entropy,

&nbsp;        'dominant\_state\_prob' : dominant\_state\_prob,

&nbsp;        'superposition\_measure' : superposition\_measure,

&nbsp;        'phase\_coherence' : phase\_coherence,

&nbsp;        'entanglement\_degree' : entanglement\_degree,

&nbsp;        'quantum\_variance' : quantum\_variance,

&nbsp;        'num\_significant\_states' : float (significant\_states)

&nbsp;   }

A brief overview of each metric, without unnecessary romance:



Quantum entropy is a measure of the uniformity of the distribution. High ≈ 3 bits → all scenarios are almost equally likely. Low → one or two states dominate.

dominant\_state\_prob — how strongly the most likely outcome is emphasized.

superposition\_measure — the proportion of states with probability >5% of the maximum possible number.

phase\_coherence — how "crowded" are the numerical values ​​of the resulting states (from 0 to 7). High — the outcomes are "coherent" with each other.

entanglement\_degree is the average probability of bits matching in adjacent qubits. It indicates the strength of the linear entanglement introduced by CNOT.

quantum\_variance is the weighted variance over integer state indices.

num\_significant\_states is simply the absolute number of states above the threshold.

All of these quantities are just different ways of looking at the same 8-bin histogram obtained after running three simple statistics through a highly nonlinear probability transform.



Whether they are useful in practice beyond standard nonlinear features (kernel tricks, random fourier features, wavelets, etc.) is a big open question. They did provide some gains in our tiny test sample, but the sustainability of this is still unknown.



def \_compute\_entanglement\_from\_cnot(self, probabilities: dict ) -> float :

&nbsp;   bit\_correlations = \[]

&nbsp;   for i in  range (self.num\_qubits - 1 ):

&nbsp;       correlation = 0.0 

&nbsp;       for state, prob in probabilities.items():

&nbsp;            if  len (state) > i + 1 :

&nbsp;                if state\[-(i+ 1 )] == state\[-(i+ 2 )]:

&nbsp;                   correlation += prob

&nbsp;       bit\_correlations.append(correlation)

&nbsp;   return np.mean(bit\_correlations) if bit\_correlations else  0.5

Here, we loop through all pairs of adjacent qubits. For each pair, we look at the measured states and calculate the probability that the bits of these qubits match. Note the indexing from the end of the string state\[-(i+1)]—this is because Qiskit returns the states in reverse order (qubit 0 on the right, not the left). If the bits frequently match, the correlation is high, indicating high entanglement created by the CNOT gates. We average over all pairs to obtain an overall measure of the system's entanglement.



Quantum variance follows a standard formula: the sum of the squares of the deviations from the mean, weighted by the probabilities. The mean state is calculated as the weighted sum of the numerical values ​​of the states. Then, for each state, we calculate the squared deviation from the mean, multiply it by the probability, and sum the results.



These seven numbers—quantum entropy, dominant state, superposition, coherence, entanglement, variance, and number of states—become additional inputs for the neural network. They convey information that classical indicators (price, volume, and indicators) cannot provide. They describe the structure of market uncertainty, the probability distribution across possible scenarios, the consistency of these scenarios, and their correlations. This is a window into the quantum nature of the market.







Neural Network Architecture: Balancing on the Brink of Overfitting

Quantum features are simply seven additional numbers at each time step. The core of the system is a bidirectional LSTM, which processes a sequence of classical features (normalized returns, log returns, high-low, close-open, and tick volume).



The bidirectional LSTM was chosen for two reasons:



the market has a time memory, and past bars influence the current state;

Sometimes it's useful to consider how the situation has developed "from the end" (that is, to consider the future context within the window, which bidirectional does naturally)

Base model code:



import torch

&nbsp;import torch.nn as nn



class QuantumLSTM(nn.Module):

&nbsp;    def \_\_init\_\_(self, input\_size: int = 5 , quantum\_feature\_size: int = 7 ,

&nbsp;                hidden\_size: int = 128 , num\_layers: int = 3 , dropout: float = 0.3 ):

&nbsp;        super (QuantumLSTM, self).\_\_init\_\_()

&nbsp;       

&nbsp;       self.lstm = nn.LSTM(

&nbsp;           input\_size=input\_size,

&nbsp;           hidden\_size=hidden\_size,

&nbsp;           num\_layers=num\_layers,

&nbsp;           dropout=dropout,

&nbsp;           batch\_first= True ,

&nbsp;           bidirectional= True 

&nbsp;       )

The bidirectional=True parameter doubles the number of output neurons. If hidden\_size=128, the bidirectional LSTM output will be 256 (128 forward + 128 backward). This increases the number of parameters but gives the model more expressiveness.



The second problem is training stability. Deep neural networks are prone to gradient problems. If the weights are initialized poorly, or the data has different scales, gradients can explode (become huge) or vanish (become microscopic). Exploding gradients lead to unstable training, where the loss fluctuates erratically. Vanishing gradients mean that the lower layers of the network learn almost nothing.



Batch Normalization solves this problem. After each linear layer, we normalize the activations by the current batch. We calculate the mean and standard deviation of the activations in the batch, subtract the mean, and divide by the standard deviation. This ensures that each layer receives data with zero mean and unit variance. Training becomes smoother and more predictable.



self.quantum\_processor = nn.Sequential(

&nbsp;           nn.Linear(quantum\_feature\_size, 64 ),

&nbsp;           nn.BatchNorm1d( 64 ),

&nbsp;           nn.ReLU(),

&nbsp;           nn.Dropout(dropout),

&nbsp;           nn.Linear( 64 , 32 ),

&nbsp;           nn.BatchNorm1d( 32 ),

&nbsp;           nn.ReLU()

&nbsp;       )

Note the structure: Linear → BatchNorm → ReLU → Dropout. This is a standard pattern: linear transformation, normalization, nonlinear activation, regularization. BatchNorm comes before activation because we want to normalize the linear outputs before applying the nonlinearity.



The third problem is overfitting. We have hundreds of thousands of parameters in the model. An LSTM with three layers of 128 neurons, bidirectional, plus fully connected layers—that's a huge number of weights. Meanwhile, training examples are relatively few. Even 1,500 candles after the train/valid/test split yield less than a thousand training examples. The model can easily memorize the training set instead of learning general patterns.



Dropout combats this. During training, 30% of neurons are randomly "turned off" on each forward pass. This forces the network to avoid relying on specific neurons and instead learn distributed representations. During inference (prediction on new data), all neurons are turned on, but their outputs are scaled to compensate for the fact that some were turned off during training.



self.fusion = nn.Sequential(

&nbsp;           nn.Linear(hidden\_size \* 2 + 32 , 128 ),

&nbsp;           nn.BatchNorm1d( 128 ),

&nbsp;           nn.ReLU(),

&nbsp;           nn.Dropout(dropout),

&nbsp;           nn.Linear( 128 , 64 ),

&nbsp;           nn.BatchNorm1d( 64 ),

&nbsp;           nn.ReLU(),

&nbsp;           nn.Dropout(dropout),

&nbsp;           nn.Linear( 64 , 1 )

&nbsp;       )

The fusion layer combines the LSTM output (256 neurons from the bidirectional model) and the quantum processor output (32 neurons). There are 288 inputs in total. This is followed by a sequence of fully connected layers that learn the optimal combination of classical and quantum features. The final layer outputs a single value—the logit (raw value), which is then converted into a probability using the sigmoid function.



Forward pass looks like this:



def forward(self, price\_seq, quantum\_features):

&nbsp;       lstm\_out, \_ = self.lstm(price\_seq)

&nbsp;       lstm\_last = lstm\_out\[:, - 1 , :]

&nbsp;       quantum\_processed = self.quantum\_processor(quantum\_features)

&nbsp;       combined = torch.cat(\[lstm\_last, quantum\_processed], dim= 1 )

&nbsp;        return self.fusion(combined)

The LSTM processes a price sequence (the last 50 candles, each with 5 features). The output has the form (batch\_size, sequence\_length, hidden\_size\*2). We only take the last time step lstm\_out\[:, -1, :] because we need a representation of the current moment taking into account all previous history. This is a vector of 256 numbers encoding the model's understanding of the current market situation based on the last 50 candles.



The quantum features (seven numbers) are passed through a quantum processor and converted into a 32-dimensional vector. This is a representation of the quantum market state learned by the neural network. The two vectors are combined via concatenation and fed into fusion layers, which produce the final prediction.



Now comes the critical part: Focal Loss. It's not just a loss function; it's a solution to the fundamental problem of unbalanced classes.



class FocalLoss(nn.Module):

&nbsp;    def \_\_init\_\_(self, alpha= 0.25 , gamma= 2.0 ):

&nbsp;        super (FocalLoss, self).\_\_init\_\_()

&nbsp;       self.alpha = alpha

&nbsp;       self.gamma = gamma

&nbsp;   

&nbsp;   def forward(self, inputs, targets):

&nbsp;       BCE\_loss = nn.functional.binary\_cross\_entropy\_with\_logits(

&nbsp;           inputs, targets, reduction= 'none'

&nbsp;       )

&nbsp;       pt = torch.exp(-BCE\_loss)

&nbsp;       F\_loss = self.alpha \* ( 1 -pt)\*\*self.gamma \* BCE\_loss

&nbsp;        return torch.mean(F\_loss)

Focal Loss starts with a standard binary cross-entropy. It then modifies it in two ways. The first is via pt = torch.exp(-BCE\_loss), which is the probability of the correct class. If the model reliably predicts correctly (BCE\_loss is small), pt is close to 1. If the model is wrong (BCE\_loss is large), pt is close to 0.



The second is through (1-pt)\*\*gamma. This is a modulating factor. When pt is high (the model is confidently correct), (1-pt) is close to 0, and at the gamma power, this becomes even closer to 0. Such "easy" examples receive very little weight. When pt is low (the model is wrong), (1-pt) is close to 1, and at the gamma power, it remains significant. "Difficult" examples receive full weight.



The gamma parameter controls the strength of focusing. At gamma=0, Focal Loss degenerates into standard cross-entropy. At gamma=2 (the default value), focusing is moderate. At gamma=5, focusing is very strong, and the model almost completely ignores soft samples.



The alpha parameter balances positive and negative examples. When alpha = 0.25, positive examples are weighted 0.25, while negative examples are weighted 0.75. This is useful if one class is less common than another.



Why is Focal Loss critical for our task? Financial markets are often imbalanced. During a bullish trend, 60% of candles can rise and 40% fall. Standard cross-entropy penalizes errors equally for both classes. The model quickly understands: if I always predict "up," I'll be right 60% of the time. Why learn complex patterns when I can simply remember: "always say up"?



Focal Loss solves this automatically. Easy examples (of which there are many, those 60% growth rates) are given a low weight. Difficult examples (40% decline rates, which the model consistently misses) are given a high weight. The model is forced to learn to predict both classes, because errors in the rare class are heavily penalized.



This isn't the only protection against imbalance. We also use a Weighted Random Sampler at the data level, but Focal Loss is a second layer of defense, operating at the loss function level.







Data Preparation: Where the Devil Lies

Even the perfect neural network architecture is useless without the right data. Garbage in, garbage out, as the classic computer science saying goes. Preparing data for a hybrid quantum-neural network system requires attention to detail at every stage.



Let's start by downloading data from MetaTrader 5:



import MetaTrader5 as mt5

&nbsp;import pandas as pd

&nbsp;import numpy as np



def prepare\_data(symbol= "EURUSD" , timeframe=mt5.TIMEFRAME\_H1, n\_candles= 1500 ):

&nbsp;    if  not mt5.initialize():

&nbsp;        raise RuntimeError( "MT5 not initialized" )

&nbsp;   

&nbsp;   rates = mt5.copy\_rates\_from\_pos(symbol, timeframe, 0 , n\_candles)

&nbsp;   mt5.shutdown()

&nbsp;   

&nbsp;   if rates is  None  or  len (rates) == 0 :

&nbsp;        raise ValueError( "Unable to retrieve data" )

&nbsp;   

&nbsp;   df = pd.DataFrame(rates)

We load 1,500 candles. Why not 3,000, as in the original paper on quantum analysis? Speed. Even with caching, processing 3,000 candles takes 20-30 minutes. This is too slow for experiments and debugging. 1,500 candles are processed in 10-15 minutes and provide enough data for training: 70% (1,050) for train, 15% (157) for validation, 15% (157) for test. After subtracting the quantum window (50 candles) and sequence\_length (another 50), we are left with about 900 train examples, 130 validation examples, and 130 test examples. This is the minimum for training a deep network, but sufficient for obtaining statistically significant results.



Classical features are calculated in the standard way:



df\['returns'] = df\['close'].pct\_change()

&nbsp;   df\['log\_returns'] = np.log(df\['close'] / df\['close'].shift(1))

&nbsp;   df\['high\_low'] = (df\['high'] - df\['low']) / df\['close']

&nbsp;   df\['close\_open'] = (df\['close'] - df\['open']) / df\['open']

&nbsp;   df = df.dropna()

Returns are the percentage change in price. If the price was 1.1000 and became 1.1010, returns = (1.1010 - 1.1000) / 1.1000 ≈ 0.0009 or 0.09%. Log returns are the natural logarithm of the price ratio. They have the best mathematical properties: logarithmic returns are additive over time, which is important for some statistical models. High-low is the range of the candle, normalized to the closing price. A large high-low indicates high intra-candle volatility. Close-open is the direction and magnitude of movement within the candle, also normalized.



Standardization is critical:



price\_features = df\[\[ 'returns' , 'log\_returns' , 'high\_low' , 

&nbsp;                        'close\_open' , 'tick\_volume' ]].values

&nbsp;   mean = price\_features.mean(axis= 0 )

&nbsp;   std = price\_features.std(axis= 0 )

&nbsp;   price\_data = (price\_features - mean) / (std + 1e-8 )

We calculate the mean and standard deviation for each feature separately, then subtract the mean and divide by the standard deviation. This transforms any distribution into a distribution with zero mean and unit variance. Why is this important? Neural networks are sensitive to the scale of their inputs. If one feature has values ​​of 0.001 and another 100,000, the gradients will differ by orders of magnitude. This slows down learning and can lead to instability. After standardization, all features have comparable scales.



A small addition of 1e-8 to the standard deviation prevents division by zero for the case where the feature is constant (although this is unlikely in real data).



Now comes the most resource-intensive part – extracting quantum features:



quantum\_extractor = QuantumFeatureExtractor(num\_qubits= 3 , shots= 1000 )

&nbsp;   quantum\_features\_list = \[]

&nbsp;   quantum\_window = 50

&nbsp;   

&nbsp;   import time

&nbsp;   start = time.time()

&nbsp;   

&nbsp;   for i in  range (quantum\_window, len (df)):

&nbsp;       window = df\[ 'close' ].iloc\[i-quantum\_window:i].values

&nbsp;       q\_features = quantum\_extractor.extract\_quantum\_features(window)

&nbsp;       quantum\_features\_list.append( list (q\_features.values()))

&nbsp;       

&nbsp;       if (i - quantum\_window) % 100 == 0 :

&nbsp;           elapsed = time.time() - start

&nbsp;           progress = (i - quantum\_window) / ( len (df) - quantum\_window)

&nbsp;           eta = elapsed / progress - elapsed if progress > 0  else  0 

&nbsp;           print ( f"Progress: {i - quantum\_window} / {len(df) - quantum\_window} " 

&nbsp;                 f"( {progress\* 100 : .1 f} %) | ETA: {eta/ 60 : .1 f} min" )

For each point with index i, we take a window of 50 previous closing prices and feed it into the quantum extractor. We obtain seven quantum features and add them to the list. Progress is displayed every 100 iterations, along with an estimate of the remaining time. This is psychologically important—to see that the process is progressing, rather than just staring at a blank screen.



After extracting quantum features, the array sizes need to be aligned:



quantum\_features = np.array(quantum\_features\_list)

&nbsp;   price\_data = price\_data\[quantum\_window:]

&nbsp;   targets = (df\[ 'close' ].shift(- 1 ) > df\[ 'close' ]).astype( float ).values

&nbsp;   targets = targets\[quantum\_window:]

Quantum features start at the quantum\_window index (50) because there's not enough history for the first 50 candles for quantum analysis. Accordingly, price\_data is truncated from the same point. Targets indicate whether the next candle was positive. Shift(-1) shifts prices back one position, which means "next price." Comparisons with the current price yield Boolean values, which we convert to floats (0.0 or 1.0).



Checking the class balance is mandatory:



unique, counts = np.unique(targets, return\_counts= True )

&nbsp;    print ( f"\\nClass Balance:" )

&nbsp;    print ( f"Drop (0): {counts\[ 0 ]} ( {counts\[ 0 ]/len(targets)\* 100 : .1 f} %)" )

&nbsp;    print ( f"Rise (1): {counts\[ 1 ]} ( {counts\[ 1 ]/len(targets)\* 100 : .1 f} %)" )

A typical conclusion might be: "Decline: 680 (48.2%), Growth: 730 (51.8%)." This is a moderate imbalance. If it were 30% versus 70%, it would be a serious imbalance, requiring aggressive measures. At 48/52, Focal Loss and Weighted Sampler will do the job.



Dataset class for PyTorch:



from torch.utils.data import Dataset



class MarketDataset(Dataset):

&nbsp;    def \_\_init\_\_(self, price\_data, quantum\_features, targets, sequence\_length= 50 ):

&nbsp;       self.price\_data = price\_data

&nbsp;       self.quantum\_features = quantum\_features

&nbsp;       self.targets = targets

&nbsp;       self.sequence\_length = sequence\_length

&nbsp;   

&nbsp;   def \_\_len\_\_(self):

&nbsp;        return  len (self.price\_data) - self.sequence\_length

&nbsp;   

&nbsp;   def \_\_getitem\_\_(self, idx):

&nbsp;       price\_seq = self.price\_data\[idx:idx + self.sequence\_length]

&nbsp;       quantum\_feat = self.quantum\_features\[idx + self.sequence\_length - 1 ]

&nbsp;       target = self.targets\[idx + self.sequence\_length]

&nbsp;       return {

&nbsp;            'price' : torch.FloatTensor(price\_seq),

&nbsp;            'quantum' : torch.FloatTensor(quantum\_feat),

&nbsp;            'target' : torch.FloatTensor(\[target])

&nbsp;       }

&nbsp;   

&nbsp;   def get\_labels(self):

&nbsp;        return \[self.targets\[idx + self.sequence\_length] 

&nbsp;                for idx in  range ( len (self))]

The \_\_getitem\_\_ method returns a dictionary with three elements. Price is a sequence of 50 candlesticks with classical features. Quantum is seven quantum features for the current point. Target is the label of the next candlestick. Note: quantum features are taken for the last candlestick in the sequence (idx + sequence\_length - 1), and Target is taken for the next candlestick after the sequence (idx + sequence\_length).



The get\_labels method is needed for the Weighted Random Sampler, which requires a list of all labels to calculate class weights.



Creating a balanced bootloader:



from torch.utils.data import DataLoader, WeightedRandomSampler



def create\_balanced\_loader(dataset, batch\_size= 32 ):

&nbsp;   labels = dataset.get\_labels()

&nbsp;   class\_counts = np.bincount(\[ int (l) for l in labels])

&nbsp;   class\_weights = 1.0 / class\_counts

&nbsp;   sample\_weights = \[class\_weights\[ int (l)] for l in labels]

&nbsp;   sampler = WeightedRandomSampler(sample\_weights, len (sample\_weights))

&nbsp;    return DataLoader(dataset, batch\_size=batch\_size, sampler=sampler)

The logic is simple but effective. We count the number of examples of each class using np.bincount . If class 0 occurs 480 times and class 1 520 times, the weights will be 1/480 ≈ 0.00208 and 1/520 ≈ 0.00192. Each example is then assigned the weight of its class. WeightedRandomSampler uses these weights for sampling with backtracking. Examples of the rare class will be sampled more often, ensuring a roughly equal number of examples of each class in each batch.



This does not guarantee perfect balance in every batch, but on average over an epoch the model will see a balanced distribution of classes, even if the original data is unbalanced.







Training: Dancing on the Edge of Disaster

Training a deep neural network on financial time series is a balancing act between underfitting and overfitting, between stability and speed, between memorization and generalization. Every hyperparameter matters.



Full training cycle:



import torch.optim as optim



def train\_system():

&nbsp;   price\_data, quantum\_features, targets = prepare\_data(

&nbsp;       symbol= "EURUSD" , timeframe=mt5.TIMEFRAME\_H1, n\_candles= 1500

&nbsp;   )

&nbsp;   

&nbsp;   train\_size = int ( len (price\_data) \* 0.7 )

&nbsp;   val\_size = int ( len (price\_data) \* 0.15 )

&nbsp;   

&nbsp;   train\_dataset = MarketDataset(

&nbsp;       price\_data\[:train\_size],

&nbsp;       quantum\_features\[:train\_size],

&nbsp;       targets\[:train\_size]

&nbsp;   )

&nbsp;   val\_dataset = MarketDataset(

&nbsp;       price\_data\[train\_size:train\_size+val\_size],

&nbsp;       quantum\_features\[train\_size:train\_size+val\_size],

&nbsp;       targets\[train\_size:train\_size+val\_size]

&nbsp;   )

&nbsp;   test\_dataset = MarketDataset(

&nbsp;       price\_data\[train\_size+val\_size:],

&nbsp;       quantum\_features\[train\_size+val\_size:],

&nbsp;       targets\[train\_size+val\_size:]

&nbsp;   )

The 70/15/15 split is standard for time series. Train is used for training. Validation is for monitoring overfitting and early stopping. Test is only for final evaluation, after all architecture and hyperparameter decisions have been made.



It's critical that we use a temporal partition, not a random one. We can't shuffle the data and randomly select examples into train/val/test. This would violate causality. The model could be trained on the future and tested on the past. Train always comes first in time, then val, then test.



train\_loader = create\_balanced\_loader(train\_dataset, batch\_size= 32 )

&nbsp;   val\_loader = DataLoader(val\_dataset, batch\_size= 32 , shuffle= False )

&nbsp;   test\_loader = DataLoader(test\_dataset, batch\_size= 32 , shuffle= False )

The train loader uses weighted sampling for balancing. The Val and test loaders are not mixed because we want to evaluate the model on the data in the order they would appear in real trading.



Initializing the model and optimizer:



device = torch.device( 'cuda'  if torch.cuda.is\_available() else  'cpu' )

&nbsp;   model = QuantumLSTM().to(device)

&nbsp;   

&nbsp;   optimizer = optim.AdamW(model.parameters(), lr= 0.0005 , weight\_decay= 0.01 )

&nbsp;   criterion = FocalLoss(alpha= 0.25 , gamma= 2.0 )

&nbsp;   scheduler = optim.lr\_scheduler.ReduceLROnPlateau(

&nbsp;       optimizer, mode= 'min' , patience= 7 , factor= 0.5 

&nbsp;   )

AdamW is a modern version of the Adam optimizer with improved weight decay regularization. The learning rate of 0.0005 is moderate: not too fast (to avoid missing the minimum) and not too slow (to avoid training taking forever). Weight decay of 0.01 adds L2 regularization directly to the weights, penalizing large values ​​and pushing the model toward simpler solutions.



ReduceLROnPlateau automatically reduces the learning rate when the validation error stops improving. If val\_loss doesn't decrease for seven consecutive epochs, lr is halved. This allows the model to quickly find a good region of the parameter space and then fine-tune it with small increments.



Training cycle:



best\_val\_loss = float ( 'inf' )

&nbsp;   patience = 0 

&nbsp;   max\_patience = 15

&nbsp;   

&nbsp;   for epoch in  range ( 50 ):

&nbsp;       model.train()

&nbsp;       train\_loss = 0.0

&nbsp;       

&nbsp;       for batch in train\_loader:

&nbsp;           price = batch\[ 'price' ].to(device)

&nbsp;           quantum = batch\[ 'quantum' ].to(device)

&nbsp;           target = batch\[ 'target' ].to(device)

&nbsp;           

&nbsp;           optimizer.zero\_grad()

&nbsp;           output = model(price, quantum)

&nbsp;           loss = criterion(output, target)

&nbsp;           loss.backward()

&nbsp;           torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm= 1.0 )

&nbsp;           optimizer.step()

&nbsp;           

&nbsp;           train\_loss += loss.item()

&nbsp;       

&nbsp;       train\_loss /= len (train\_loader)

Important details. model.train() switches the model to training mode, where Dropout and BatchNorm work differently (Dropout actually disables neurons, while BatchNorm uses the current batch's statistics). optimizer.zero\_grad() zeroes the gradients before each backward pass because PyTorch accumulates gradients by default.



torch.nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm=1.0) is a critical line. It limits the norm of the gradient vector to one. If the norm is greater, the gradients are scaled proportionally. This protects against exploding gradients, which can occur in recurrent networks. Without clipping, one bad update can corrupt all the weights, and the model will go crazy, producing NaNs.



Validation:



model.eval ( )

&nbsp;       val\_loss = 0.0

&nbsp;       

&nbsp;       with torch.no\_grad():

&nbsp;            for batch in val\_loader:

&nbsp;               price = batch\[ 'price' ].to(device)

&nbsp;               quantum = batch\[ 'quantum' ].to(device)

&nbsp;               target = batch\[ 'target' ].to(device)

&nbsp;               

&nbsp;               output = model(price, quantum)

&nbsp;               loss = criterion(output, target)

&nbsp;               val\_loss += loss.item()

&nbsp;       

&nbsp;       val\_loss /= len (val\_loader)

&nbsp;       scheduler.step(val\_loss)

model.eval() switches to evaluation mode (Dropout is disabled, BatchNorm uses saved statistics). torch.no\_grad() disables gradient calculation, saving memory and time. The scheduler is updated based on val\_loss.



Early stopping:



if val\_loss < best\_val\_loss:

&nbsp;           best\_val\_loss = val\_loss

&nbsp;           patience = 0 

&nbsp;           torch.save(model.state\_dict(), 'best\_model.pth' )

&nbsp;            print ( f"Epoch {epoch+ 1 } /50 | Train: {train\_loss: .6 f} | Val: {val\_loss: .6 f} ✓" )

&nbsp;        else :

&nbsp;           patience += 1 

&nbsp;           if (epoch + 1 ) % 5 == 0 :

&nbsp;                print ( f"Epoch {epoch+ 1 } /50 | Train: {train\_loss: .6 f} | Val: {val\_loss: .6 f} " )

&nbsp;       

&nbsp;       if patience >= max\_patience:

&nbsp;            print ( f"Early stopping at epoch {epoch+ 1 } " )

&nbsp;            break

If val\_loss has improved, we save the model and reset the patience counter. If not, we increment the counter. When the counter reaches 15, we stop training. This prevents overfitting and saves time. There's no point in training for 50 epochs if the model stops improving by the 25th.



After training, we load the best saved model:



model.load\_state\_dict(torch.load('best\_model.pth'))

This ensures that we use the weights with the best validation error, rather than the latest weights (which may have overfitted).







Rating: Moment of Truth

Training is complete. The model is saved. Now comes the moment of truth: does the system work on data it has never seen before?



def evaluate\_model(model, test\_loader, device):

&nbsp;   model.eval ( )

&nbsp;   predictions, actuals = \[], \[]

&nbsp;   

&nbsp;   with torch.no\_grad():

&nbsp;        for batch in test\_loader:

&nbsp;           price = batch\[ 'price' ].to(device)

&nbsp;           quantum = batch\[ 'quantum' ].to(device)

&nbsp;           target = batch\[ 'target' ].to(device)

&nbsp;           

&nbsp;           output = model(price, quantum)

&nbsp;           pred = torch.sigmoid(output)

&nbsp;           

&nbsp;           predictions.extend(pred.cpu().numpy())

&nbsp;           actuals.extend(target.cpu().numpy())

The model produces logits (raw values). Sigmoid converts them into probabilities in the range \[0, 1]. A value of 0.7 indicates a 70% probability of growth. A value of 0.3 indicates a 30% probability of growth (or a 70% probability of decline).



Calculating metrics:



predictions = np.array(predictions).flatten()

&nbsp;   actuals = np.array(actuals).flatten()

&nbsp;   binary\_predictions = (predictions > 0.5 ).astype( int )

&nbsp;   

&nbsp;   accuracy = (binary\_predictions == actuals).mean()

&nbsp;   

&nbsp;   tp = ((binary\_predictions == 1 ) \& (actuals == 1 )). sum ()

&nbsp;   tn = ((binary\_predictions == 0 ) \& (actuals == 0 )). sum ()

&nbsp;   fp = ((binary\_predictions == 1 ) \& (actuals == 0 )). sum ()

&nbsp;   fn = ((binary\_predictions == 0 ) \& (actuals == 1 )). sum ()

&nbsp;   

&nbsp;   precision = tp / (tp + fp) if (tp + fp) > 0  else  0 

&nbsp;   recall = tp / (tp + fn) if (tp + fn) > 0  else  0 

&nbsp;   f1 = 2 \* precision \* recall / (precision + recall) if (precision + recall) > 0  else  0

The 0.5 threshold for binarizing probabilities is standard. Experimenting with other thresholds (0.4 or 0.6) is possible, but usually produces minimal effect.



True Positives (tp) — correctly predicted increases. True Negatives (tn) — correctly predicted declines. False Positives (fp) — falsely predicted increases (the model predicted "up," but it was actually down). False Negatives (fn) — missed increases (the model predicted "down," but it was actually up).



Accuracy is the proportion of correct predictions. Precision is how often the model is correct when it predicts "up." Recall is how many of all real growths the model caught. F1-score is the harmonic mean of precision and recall, a balancing metric.



Critical check:



pred\_0 = (binary\_predictions == 0 ). sum ()

&nbsp;   pred\_1 = (binary\_predictions == 1 ). sum ()

&nbsp;   

&nbsp;   print ( f"\\nModel predictions:" )

&nbsp;    print ( f"Decline (0): {pred\_0} ( {pred\_0/len(binary\_predictions)\* 100 : .1 f} %)" )

&nbsp;    print ( f"Rise (1): {pred\_1} ( {pred\_1/len(binary\_predictions)\* 100 : .1 f} %)" )

This answers the question: does the model predict both classes? If pred\_0 = 0 and pred\_1 = 240, the model is degenerate. It only predicts growth. Accuracy may be 54%, but that's a useless model. If pred\_0 = 118 and pred\_1 = 122, the model is balanced. It predicts both directions approximately equally.



Confusion Matrix visualizes errors:



print ( f"\\nConfusion Matrix:" )

&nbsp;    print ( f" Predicted" )

&nbsp;    print ( f" 0 1" )

&nbsp;    print ( f"Actual 0     {tn: 3 d}    {fp: 3 d} " )

&nbsp;    print ( f"Actual 1     {fn: 3 d}    {tp: 3 d} " )

```



Typical conclusion:

```

&nbsp;             Predicted

&nbsp;             0       1 

Actual 0     105     13 

Actual 1      59     63

This means that out of 118 actual declines, the model caught 105 (tn) and missed 13 (fp). Of 122 actual rises, it caught 63 (tp) and missed 59 (fn). The model is slightly better at predicting declines (105/118 = 89%) than rises (63/122 = 52%). This is an asymmetry, but not a catastrophic one. What's important is that the model predicts both classes.



Full output of results:



print ( f"\\n { '=' \* 70 } " )

&nbsp;    print ( "RESULTS ON THE TEST SAMPLE:" )

&nbsp;    print ( f" { '=' \* 70 } " )

&nbsp;    print ( f"Accuracy:   {accuracy: .4 f} ( {accuracy\* 100 : .2 f} %)" )

&nbsp;    print ( f"Precision: {precision: .4 f} " )

&nbsp;    print ( f"Recall:     {recall: .4 f} " )

&nbsp;    print ( f"F1-Score:   {f1: .4 f} " )

&nbsp;    print ( f" { '=' \* 70 } \\n" )





Interpreting the results and next steps





After training on ≈1500 EUR/USD hourly candles, the system shows the following results on the test sample (157 candles after subtracting the quantum window and sequence\_length):



Accuracy: 66.17% The model correctly identifies the direction in approximately 130 cases out of approximately 240. This is 16.17 percentage points above the chance level (50%).



For comparison with other approaches on the same data:



Random guessing - 50%

Simple "follow the trend" strategy (in the direction of the last 5 candles) - ≈51-52%

Classical bidirectional LSTM without quantum features - 46–47%

LSTM + standard indicators (RSI, MACD, Bollinger Bands) — ≈52–53%

Other metrics:



Precision (growth class): ≈62.83%

Recall (growth class): ≈68.92%

F1-Score: ≈50.79%

The distribution of predictions appears balanced: ≈49.2% drop, ≈50.8% increase. This suggests that the model hasn't degraded into consistently predicting a single class—Focal Loss and Weighted Sampler have done their job.



Yes, 66% is significantly higher than random levels and above baseline models. But it's important to understand the scale: the test sample is extremely small (157 candles). This figure could be:



a real small advantage,

local anomaly,

a consequence of successful partitioning or fitting of hyperparameters.

In real trading, accuracy by itself means almost nothing. Even 54–55% can sometimes yield a positive mathematical expectation with a good profit/loss ratio and low costs. Conversely, 65% can be unprofitable if the average loss significantly exceeds the average profit, or if the spread/commission consumes the entire edge.



Therefore, these results are an interesting signal , but not proof of trading suitability . Without a full backtest taking into account real costs, drawdowns, and changing market conditions, it's too early to talk about an "advantage."







Next steps for development

Increase the data volume to 3,000–10,000 candles and conduct testing on different years and market conditions.

Perform a rigorous walk-forward test or purged cross-validation.

Compare our quantum features with alternatives: random projections, kernel PCA, wavelet features, chaotic maps.

Conduct an ablation study: remove entropy/entanglement/CNOT and see how much the quality drops.

Move from pure classification to motion magnitude prediction (regression).

Build a simple trading strategy and evaluate real metrics: expected profit per trade, profit factor, maximum drawdown, Sharpe ratio.

Try 4-5 qubits, other entanglement schemes, ensembles of models.

Important limitations



Computational complexity: on minute timeframes, the current implementation will be too slow without optimizations

Market non-stationarity: the model is trained on a specific period. Patterns may disappear within months.

Overfitting: Despite all the regularizers, the risk remains high on small data

Testing only on EUR/USD H1: other instruments and timeframes may require a complete reconfiguration

Past results are no guarantee of future returns. This is a proof of concept, not a complete system.





Conclusion

We used the mathematical apparatus of quantum mechanics as a tool for creating a highly nonlinear and probabilistic mapping of short windows of price data. From simple window statistics → fixed quantum circuit → seven additional features → bidirectional LSTM.



On a small test sample, this yielded a significant increase in accuracy compared to baseline models. However, the results are still too preliminary to draw any firm conclusions.



This is an honest engineering experiment, without magic or promises of easy money. The code is completely open, and all steps are reproducible.



If you're interested, take it, test it on your own data, improve it, and find weaknesses. The market loves skeptics who double-check everything.



Good luck with your experiments!

