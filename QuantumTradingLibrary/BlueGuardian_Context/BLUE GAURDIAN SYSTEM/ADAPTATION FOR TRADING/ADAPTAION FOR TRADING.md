# &nbsp;             ADAPTAION FOR TRADING













Evgeniy Koshtenko

The last year and a half of working with large language models in trading has revealed a paradoxical problem: the more accurately a model predicts the market today, the faster it degrades tomorrow. This isn't a theoretical observation from academic journals—it's a reality I've witnessed in a live trading account, where a model with 73% accuracy on Monday showed 51% by Friday. The reason is simple: markets change faster than we can retrain models.





A problem that no one is solving

When I first started using Llama 3.2 to forecast currency pairs, the process seemed elegant: collect three months of historical data, fine-tune the model on 2,000 examples, and get excellent results. After two weeks, the model starts to fail. It's not catastrophic—it's just that confidence drops, accuracy devolves into randomness, and the worst part: the model continues to be confident in its predictions, even though they're no longer working.



The classic solution is to retrain the model on fresh data. It sounds logical until you start doing the math. Fine-tuning Llama 3.2:3b on 2,000 examples takes about 40 minutes on an RTX 3090. If we do this every week, we'll end up with 160 minutes of pure downtime per month. Add in data preparation, validation, and testing—that's half a day's work. And that's assuming we even notice any significant model degradation.



But the main problem isn't time. The main problem is that when retraining, the model forgets old patterns. The market is cyclical: what didn't work the last two weeks may return in a month. Standard fine-tuning works on the principle of rewriting: new knowledge displaces old. We end up with a model that works perfectly in the current market regime and is completely helpless when the regime changes.





The SEAL Concept: Learning Without Forgetting

Somewhere between another failed experiment with daily retraining and reading articles about continuous learning, I realized a simple thing: a model should learn like a human—not replace old knowledge with new, but supplement it. When an experienced trader sees a new pattern, they don't forget the old ones—they add the new one to their arsenal and begin to understand the conditions under which each pattern works.



This is how the SEAL concept—Self-Evolving Adaptive Learning—was born. It's not just scheduled fine-tuning, but continuous model evolution based on real trading results. Every trade becomes a learning experience, and every result, feedback. The model doesn't just predict price movements—it learns from its mistakes and successes.



I wrote the first prototype overnight. Conceptually, it all looked simple:



class SEALSystem:

&nbsp;    def \_\_init\_\_(self, model\_name: str ):

&nbsp;       self.model\_name = model\_name

&nbsp;       self.trade\_memory = \[]

&nbsp;       self.learning\_buffer = \[]

&nbsp;   

&nbsp;   def record\_trade(self, prediction: dict , outcome: dict ):

&nbsp;        """Record the outcome of the trade"""

&nbsp;       example = {

&nbsp;           'input' : self.\_format\_input(prediction),

&nbsp;            'output' : self.\_format\_output(outcome),

&nbsp;            'timestamp' : time.time()

&nbsp;       }

&nbsp;       self.learning\_buffer.append(example)

Seems trivial, right? But the devil, as always, is in the details. The first run revealed a fundamental problem: how to distinguish a good pattern from random luck? If a model correctly predicted a 50-pip move up in EUR/USD, did it identify a real pattern, or simply miscalculated amid high volatility?





Memory: All that glitters is not gold

Human memory is selective for a reason—we remember significant events and forget routine ones. Similarly, the SEAL system had to learn to distinguish training examples based on their value. Not every transaction is equally valuable for learning.



I added a weighting system for examples:



def calculate\_example\_weight(self, prediction: dict , outcome: dict ) -> float :

&nbsp;    """Calculate the weight of the training example"""

&nbsp;   

&nbsp;   # Base weight - prediction accuracy 

&nbsp;   predicted\_direction = prediction\[ 'direction' ]

&nbsp;   actual\_direction = 'UP'  if outcome\[ 'profit' ] > 0  else  'DOWN' 

&nbsp;   base\_weight = 1.0  if predicted\_direction == actual\_direction else  0.3

&nbsp;   

&nbsp;   # Confidence modifier: the more confident the model was, the more important the result is 

&nbsp;   confidence = prediction\[ 'confidence' ] / 100.0 

&nbsp;   confidence\_modifier = 1.0 + (confidence - 0.5 )

&nbsp;   

&nbsp;   # Movement magnitude modifier: strong movements are more important than weak ones 

&nbsp;   pips = abs (outcome\[ 'pips' ])

&nbsp;   movement\_modifier = min (pips / 50.0 , 2.0 )

&nbsp;   

&nbsp;   # Rarity modifier: rare features are more important than frequent ones 

&nbsp;   market\_regime = self.\_classify\_market\_regime(prediction\[ 'features' ])

&nbsp;   rarity\_modifier = self.\_get\_regime\_rarity(market\_regime)

&nbsp;   

&nbsp;   weight = base\_weight \* confidence\_modifier \* movement\_modifier \* rarity\_modifier

&nbsp;   return weight

This formula wasn't born from mathematical research, but from observations of real trades. The first version simply calculated correct and incorrect predictions. But it quickly became clear that the model was memorizing flat patterns (which are the majority) and losing track of trends. Adding movement\_modifier solved the problem—strong movements now carried more weight, forcing the model to remember their patterns.



The confidence modifier emerged from an analysis of false positives. It turned out that when the model was 95% confident and then wrong, it was a critical training example. It signals that the model is seeing a pattern where there isn't one, and it's these examples that need to be memorized first.





Memory architecture: Prioritized circular buffer

We can't simply lump all the examples together and retrain the model on the entire history. Firstly, it's computationally expensive; secondly, old examples may no longer be relevant to the current market; thirdly, we need a balance between data freshness and historical context.



The solution came from an unexpected source—operating system architecture. Remember the virtual memory page replacement algorithms? I adapted a combination of LRU (Least Recently Used) and priority eviction:



class PriorityMemoryBuffer:

&nbsp;    def \_\_init\_\_(self, max\_size: int = 1000 ):

&nbsp;       self.max\_size = max\_size

&nbsp;       self.buffer = \[]

&nbsp;       self.weights = \[]

&nbsp;       self.timestamps = \[]

&nbsp;   

&nbsp;   def add(self, example: dict , weight: float ):

&nbsp;       timestamp = time.time()

&nbsp;       

&nbsp;       if  len (self.buffer) < self.max\_size:

&nbsp;           self.buffer.append(example)

&nbsp;           self.weights.append(weight)

&nbsp;           self.timestamps.append(timestamp)

&nbsp;       else :

&nbsp;            # Find a candidate for displacement

&nbsp;           scores = self.\_calculate\_retention\_scores()

&nbsp;           min\_idx = np.argmin(scores)

&nbsp;           

&nbsp;           # Replace only if the new example is more important 

&nbsp;           if weight > self.weights\[min\_idx]:

&nbsp;               self.buffer\[min\_idx] = example

&nbsp;               self.weights\[min\_idx] = weight

&nbsp;               self.timestamps\[min\_idx] = timestamp

&nbsp;   

&nbsp;   def \_calculate\_retention\_scores(self) -> np.ndarray:

&nbsp;        """Calculate the retention scores of an example."""

&nbsp;       current\_time = time.time()

&nbsp;       

&nbsp;       # Normalize weights and age 

&nbsp;       norm\_weights = np.array(self.weights) / max (self.weights)

&nbsp;       ages = current\_time - np.array(self.timestamps)

&nbsp;       norm\_ages = 1.0 - (ages / max (ages))   # Invert: fresher = more important

&nbsp;       

&nbsp;       # Combined score: 70% sample weight, 30% freshness 

&nbsp;       scores = 0.7 \* norm\_weights + 0.3 \* norm\_ages

&nbsp;        return scores

This system solves several problems simultaneously: old but important examples (for example, rare patterns with strong movements) are retained longer; new examples are buffered more easily, even if they are not super important (the market changes, and we need current context); mediocre examples of average age are displaced first.





Incremental Learning: When to Fine-Tune

A naive implementation would trigger fine-tuning after every closed trade. This is madness—we would end up with a system that constantly learns and never trades. A trigger was needed that balances model relevance with computational costs.



The first version used a simple counter - every 50 transactions:



def record\_trade(self, prediction: dict , outcome: dict ):

&nbsp;   weight = self.calculate\_example\_weight(prediction, outcome)

&nbsp;   self.memory.add(self.\_create\_example(prediction, outcome), weight)

&nbsp;   

&nbsp;   self.total\_trades += 1

&nbsp;   

&nbsp;   if self.total\_trades % 50 == 0 :

&nbsp;       log.info( f"SEAL: Accumulated {self.total\_trades} trades - starting additional training..." )

&nbsp;       self.\_trigger\_finetuning()

This worked, but it was inefficient. During calm periods, 50 trades could accumulate over weeks, and during volatile periods, in a day. The model either became outdated or overfitted on too short a period.



A smarter version analyzes the quality of predictions:



def should\_trigger\_learning(self) -> bool :

&nbsp;    """Determine the need for additional training"""

&nbsp;   

&nbsp;   # The minimum threshold is at least 30 new examples 

&nbsp;   if  len (self.learning\_buffer) < 30 :

&nbsp;        return  False

&nbsp;   

&nbsp;   # Analyze the last 20 trades 

&nbsp;   recent\_predictions = self.get\_recent\_predictions( 20 )

&nbsp;    if  len (recent\_predictions) < 20 :

&nbsp;        return  False

&nbsp;   

&nbsp;   # Calculate the current accuracy 

&nbsp;   correct = sum ( 1  for p in recent\_predictions if p\[ 'correct' ])

&nbsp;   accuracy = correct / len (recent\_predictions)

&nbsp;   

&nbsp;   # Trigger 1: accuracy dropped below 55% 

&nbsp;   if accuracy < 0.55 :

&nbsp;       log.warning( f"SEAL: Accuracy {accuracy: .1 %} - requires further training" )

&nbsp;        return  True

&nbsp;   

&nbsp;   # Trigger 2: many examples have accumulated (>100) 

&nbsp;   if  len (self.learning\_buffer) > 100 :

&nbsp;       log.info( f"SEAL: Accumulated {len(self.learning\_buffer)} examples" )

&nbsp;        return  True

&nbsp;   

&nbsp;   # Trigger 3: New market regime detected 

&nbsp;   if self.\_detect\_regime\_shift():

&nbsp;       log.warning( "SEAL: Market Regime Change - Adaptation" )

&nbsp;        return  True

&nbsp;   

&nbsp;   return  False

The regime shift detector deserves special attention. I used a combination of volatility, volume, and model error distribution:

def \_detect\_regime\_shift(self) -> bool :

&nbsp;    """Detects a market regime shift"""

&nbsp;   

&nbsp;   recent = self.get\_recent\_predictions( 30 )

&nbsp;    if  len (recent) < 30 :

&nbsp;        return  False

&nbsp;   

&nbsp;   # Analyze the distribution of errors 

&nbsp;   errors = \[ abs (p\[ 'predicted\_pips' ] - p\[ 'actual\_pips' ]) for p in recent]

&nbsp;   

&nbsp;   # Compare with the historical average

&nbsp;   historical\_error = self.get\_historical\_average\_error()

&nbsp;   current\_error = np.mean(errors)

&nbsp;   

&nbsp;   # Mode change = error increased by 50%+ 

&nbsp;   if current\_error > historical\_error \* 1.5 :

&nbsp;        return  True

&nbsp;   

&nbsp;   # Check the change in volatility 

&nbsp;   current\_volatility = np.std(\[p\[ 'actual\_pips' ] for p in recent])

&nbsp;   historical\_volatility = self.get\_historical\_volatility()

&nbsp;   

&nbsp;   if  abs (current\_volatility - historical\_volatility) / historical\_volatility > 0.4 :

&nbsp;        return  True

&nbsp;   

&nbsp;   return  False



Forming training examples: context over details

When I started generating examples for further training, the first version looked like JSON with raw data:



{

&nbsp;  "RSI" : 45.3 ,

&nbsp;  "MACD" : - 0.0012 ,

&nbsp;  "price" : 1.0856 ,

&nbsp;  "direction" : "UP" 

}

The model trained, but the results were mediocre. The problem was that LLM was trained to work with natural language, not tabular data. The problem needed to be reformulated to leverage the strengths of the language model—its understanding of context and relationships.



The new format became narrative:



def \_create\_learning\_example(self, prediction: dict , outcome: dict ) -> dict :

&nbsp;    """Create a training example in natural language format"""

&nbsp;   

&nbsp;   features = prediction\[ 'features' ]

&nbsp;   

&nbsp;   # We form a contextual description of the market situation

&nbsp;   context\_parts = \[]

&nbsp;   

&nbsp;   # Trend 

&nbsp;   if features\[ 'EMA\_50' ] > features\[ 'EMA\_200' ]:

&nbsp;       trend = "uptrend (EMA50 > EMA200)" 

&nbsp;   else :

&nbsp;       trend = "downtrend (EMA50 < EMA200)" 

&nbsp;   context\_parts.append( f"Market is in a {trend} state " )

&nbsp;   

&nbsp;   # Overbought/oversold 

&nbsp;   rsi = features\[ 'RSI' ]

&nbsp;    if rsi > 70 :

&nbsp;       context\_parts.append( f"RSI= {rsi: .1 f} indicates overbought" )

&nbsp;    elif rsi < 30 :

&nbsp;       context\_parts.append( f"RSI= {rsi: .1 f} indicates oversold" )

&nbsp;    else :

&nbsp;       context\_parts.append( f"RSI= {rsi: .1 f} in neutral zone" )

&nbsp;   

&nbsp;   # Volatility 

&nbsp;   bb\_position = features\[ 'BB\_position' ]

&nbsp;    if bb\_position > 0.8 :

&nbsp;       context\_parts.append( "price at the upper Bollinger band" )

&nbsp;    elif bb\_position < 0.2 :

&nbsp;       context\_parts.append( "price at the lower Bollinger band" )

&nbsp;   

&nbsp;   # Quantum features 

&nbsp;   if  'quantum\_entropy'  in features:

&nbsp;       entropy = features\[ 'quantum\_entropy' ]

&nbsp;        if entropy > 6.0 :

&nbsp;           context\_parts.append( "quantum entropy is high (uncertainty)" )

&nbsp;        elif entropy < 4.0 :

&nbsp;           context\_parts.append( "quantum entropy is low (certainty)" )

&nbsp;   

&nbsp;   context = ", " .join(context\_parts) + "."

&nbsp;   

&nbsp;   # Generate the result 

&nbsp;   actual\_direction = "up"  if outcome\[ 'profit' ] > 0  else  "down" 

&nbsp;   pips = abs (outcome\[ 'pips' ])

&nbsp;   

&nbsp;   if outcome\[ 'correct' ]:

&nbsp;       result = f"Price moved {actual\_direction} by {pips: .1 f} pips, as predicted." 

&nbsp;   else :

&nbsp;       predicted\_dir = "up"  if prediction\[ 'direction' ] == 'UP'  else  "down" 

&nbsp;       result = f"The prediction was {predicted\_dir} , but the price went {actual\_direction} by {pips: .1 f} pips."

&nbsp;   

&nbsp;   return {

&nbsp;        "prompt" : f"Analysis {prediction\[ 'symbol' ]} : {context} " ,

&nbsp;        "completion" : result,

&nbsp;        "weight" : self.calculate\_example\_weight(prediction, outcome)

&nbsp;   }

This change resulted in an 8% increase in accuracy. The model began to understand the relationships between indicators, rather than simply memorizing numbers. It learned to see that overbought RSI in an uptrend is not the same as overbought in a flat.





Practical implementation: integration into the trading system

It all sounds great in theory, but the real test is integration into a real trading system. SEAL wasn't meant to be a standalone module with a life of its own. It was meant to be a natural part of the trading cycle.



Here's what the full cycle looks like in my system:



class QuantumFusionTrader:

&nbsp;    def \_\_init\_\_(self):

&nbsp;       self.catboost\_model = CatBoostClassifier()

&nbsp;       self.catboost\_model.load\_model( "models/catboost\_quantum\_3d.cbm" )

&nbsp;       self.quantum\_encoder = QuantumEncoder(n\_qubits= 8 , n\_shots= 2048 )

&nbsp;       self.seal = SEALSystem(model\_name= "koshtenco/quantum-trader-fusion-3d" )

&nbsp;       self.active\_trades = {}

&nbsp;   

&nbsp;   def analyze\_and\_trade(self, symbol: str ):

&nbsp;        """Full cycle of analysis and trading"""

&nbsp;       

&nbsp;       # 1. Obtaining data

&nbsp;       df = self.load\_symbol\_data(symbol)

&nbsp;       features = self.calculate\_features(df)

&nbsp;       

&nbsp;       # 2. Quantum coding

&nbsp;       quantum\_features = self.quantum\_encoder.encode\_and\_measure(

&nbsp;           features.iloc\[- 1 ].values

&nbsp;       )

&nbsp;       

&nbsp;       # 3. CatBoost prediction

&nbsp;       catboost\_pred = self.catboost\_model.predict\_proba(

&nbsp;           features.iloc\[- 1 :].values

&nbsp;       )\[ 0 ]

&nbsp;       catboost\_confidence = max (catboost\_pred) \* 100 

&nbsp;       catboost\_direction = 'UP'  if catboost\_pred\[ 1 ] > 0.5  else  'DOWN'

&nbsp;       

&nbsp;       #4. LLM analysis taking into account SEAL experience

&nbsp;       llm\_response = self.get\_llm\_prediction(

&nbsp;           symbol, features.iloc\[- 1 ], quantum\_features,

&nbsp;           catboost\_direction, catboost\_confidence

&nbsp;       )

&nbsp;       

&nbsp;       # 5. Final decision

&nbsp;       final\_decision = self.combine\_predictions(

&nbsp;           catboost\_pred, llm\_response

&nbsp;       )

&nbsp;       

&nbsp;       # 6. Open a deal 

&nbsp;       if final\_decision\[ 'confidence' ] >= MIN\_CONFIDENCE:

&nbsp;           ticket = self.open\_trade(symbol, final\_decision)

&nbsp;           

&nbsp;           # Save context for SEAL

&nbsp;           self.active\_trades\[ticket] = {

&nbsp;               'symbol' : symbol,

&nbsp;                'prediction' : final\_decision,

&nbsp;                'features' : features.iloc\[- 1 ].to\_dict(),

&nbsp;                'quantum\_features' : quantum\_features,

&nbsp;                'open\_time' : time.time(),

&nbsp;                'open\_price' : self.get\_current\_price(symbol)

&nbsp;           }

&nbsp;   

&nbsp;   def on\_trade\_closed(self, ticket: int , close\_price: float , profit: float ):

&nbsp;        """Processing the trade closing"""

&nbsp;       

&nbsp;       if ticket not  in self.active\_trades:

&nbsp;            return

&nbsp;       

&nbsp;       trade\_data = self.active\_trades\[ticket]

&nbsp;       

&nbsp;       # Calculate the result

&nbsp;       pips = self.calculate\_pips(

&nbsp;           trade\_data\[ 'open\_price' ],

&nbsp;           close\_price,

&nbsp;           trade\_data\[ 'symbol' ]

&nbsp;       )

&nbsp;       

&nbsp;       correct = (profit > 0  and trade\_data\[ 'prediction' ]\[ 'direction' ] == 'UP' ) or \\

&nbsp;                 (profit < 0  and trade\_data\[ 'prediction' ]\[ 'direction' ] == 'DOWN' )

&nbsp;       

&nbsp;       outcome = {

&nbsp;           'close\_price' : close\_price,

&nbsp;            'profit' : profit,

&nbsp;            'pips' : pips,

&nbsp;            'correct' : correct,

&nbsp;            'duration' : time.time() - trade\_data\[ 'open\_time' ]

&nbsp;       }

&nbsp;       

&nbsp;       # SEAL records the result 

&nbsp;       self.seal.record\_trade(trade\_data\[ 'prediction' ], outcome)

&nbsp;       

&nbsp;       # Checking the need for additional training 

&nbsp;       if self.seal.should\_trigger\_learning():

&nbsp;           self.trigger\_seal\_learning()

&nbsp;       

&nbsp;       del self.active\_trades\[ticket]

A critical point is that SEAL is triggered asynchronously . We don't block trading during the training period. When the trigger is triggered, I run fine-tuning in the background:



def trigger\_seal\_learning(self):

&nbsp;    """Trigger sealed learning asynchronously."""

&nbsp;   

&nbsp;   examples = self.seal.prepare\_learning\_dataset()

&nbsp;   

&nbsp;   if  len (examples) < 30 :

&nbsp;       log.warning( "SEAL: Not enough examples for training" )

&nbsp;        return

&nbsp;   

&nbsp;   # Save in JSONL 

&nbsp;   dataset\_path = f"seal\_datasets/iteration\_ {self.seal.iteration} .jsonl" 

&nbsp;   with  open (dataset\_path, 'w' , encoding= 'utf-8' ) as f:

&nbsp;        for ex in examples:

&nbsp;           f.write(json.dumps(ex, ensure\_ascii= False ) + '\\n' )

&nbsp;   

&nbsp;   # Run ollama finetune in the background 

&nbsp;   modelfile\_content = f"""

FROM {self.seal.model\_name} 

ADAPTER {dataset\_path}

PARAMETER temperature 0.7

PARAMETER top\_p 0.9

"""

&nbsp;   

&nbsp;   modelfile\_path = f"seal\_models/Modelfile\_ {self.seal.iteration} " 

&nbsp;   with  open (modelfile\_path, 'w' ) as f:

&nbsp;       f.write(modelfile\_content)

&nbsp;   

&nbsp;   # Asynchronous run 

&nbsp;   new\_model\_name = f" {self.seal.model\_name} -seal- {self.seal.iteration} "

&nbsp;   

&nbsp;   process = subprocess.Popen(

&nbsp;       \[ 'ollama' , 'create' , new\_model\_name, '-f' , modelfile\_path],

&nbsp;       stdout=subprocess.PIPE,

&nbsp;       stderr=subprocess.PIPE

&nbsp;   )

&nbsp;   

&nbsp;   log.info( f"SEAL: Retraining started → {new\_model\_name} " )

&nbsp;   

&nbsp;   # Don't wait for completion - trading continues 

&nbsp;   # The new model will be used after training is completed

&nbsp;   self.seal.pending\_model = new\_model\_name

&nbsp;   self.seal.iteration += 1



Monitoring Evolution: How to Tell if SEAL Is Working

The biggest danger of adaptive systems is unnoticeable degradation. The model can learn, but it can learn the wrong things. I needed a monitoring system that would show not just the win rate, but the direction of evolution.



I built a metrics tracker with time analysis:



class SEALMetricsTracker:

&nbsp;    def \_\_init\_\_(self):

&nbsp;       self.metrics\_history = \[]

&nbsp;       self.window\_size = 100   # Analyze the last 100 transactions

&nbsp;   

&nbsp;   def add\_trade\_result(self, prediction: dict , outcome: dict ):

&nbsp;        """Adding the trade result"""

&nbsp;       

&nbsp;       metrics = {

&nbsp;           'timestamp' : time.time(),

&nbsp;            'correct' : outcome\[ 'correct' ],

&nbsp;            'confidence' : prediction\[ 'confidence' ],

&nbsp;            'pips' : outcome\[ 'pips' ],

&nbsp;            'profit' : outcome\[ 'profit' ],

&nbsp;            'model\_version' : self.current\_model\_version

&nbsp;       }

&nbsp;       

&nbsp;       self.metrics\_history.append(metrics)

&nbsp;       

&nbsp;       # Periodic analysis 

&nbsp;       if  len (self.metrics\_history) % self.window\_size == 0 :

&nbsp;           self.analyze\_evolution()

&nbsp;   

&nbsp;   def analyze\_evolution(self):

&nbsp;        """Analyze the evolution of the model"""

&nbsp;       

&nbsp;       if  len (self.metrics\_history) < self.window\_size \* 2 :

&nbsp;            return

&nbsp;       

&nbsp;       # Take two consecutive windows

&nbsp;       recent = self.metrics\_history\[-self.window\_size:]

&nbsp;       previous = self.metrics\_history\[-self.window\_size\* 2 :-self.window\_size]

&nbsp;       

&nbsp;       # Compare key metrics 

&nbsp;       recent\_accuracy = sum ( 1  for t in recent if t\[ 'correct' ]) / len (recent)

&nbsp;       previous\_accuracy = sum ( 1  for t in previous if t\[ 'correct' ]) / len (previous)

&nbsp;       

&nbsp;       recent\_profit = sum (t\[ 'profit' ] for t in recent)

&nbsp;       previous\_profit = sum (t\[ 'profit' ] for t in previous)

&nbsp;       

&nbsp;       # Confidence Calibration

&nbsp;       recent\_calibration = self.\_calculate\_calibration(recent)

&nbsp;       previous\_calibration = self.\_calculate\_calibration(previous)

&nbsp;       

&nbsp;       log.info( f"SEAL EVOLUTION:" )

&nbsp;       log.info( f" Accuracy: {previous\_accuracy: .1 %} → {recent\_accuracy: .1 %} " +

&nbsp;                f"( {self.\_format\_delta(recent\_accuracy - previous\_accuracy)} )" )

&nbsp;       log.info( f" PnL: {previous\_profit: .2 f} → {recent\_profit: .2 f} " +

&nbsp;                f"( {self.\_format\_delta(recent\_profit - previous\_profit)} )" )

&nbsp;       log.info( f" Calibration: {previous\_calibration: .3 f} → {recent\_calibration: .3 f} " +

&nbsp;                f"( {self.\_format\_delta(recent\_calibration - previous\_calibration)} )" )

&nbsp;   

&nbsp;   def \_calculate\_calibration(self, trades: list ) -> float :

&nbsp;        """Calculate the quality of the confidence calibration"""

&nbsp;       

&nbsp;       # Group trades by confidence level 

&nbsp;       bins = \[ 0 , 60 , 70 , 80 , 90 , 100 ]

&nbsp;       calibration\_error = 0

&nbsp;       

&nbsp;       for i in  range ( len (bins)- 1 ):

&nbsp;           bin\_trades = \[t for t in trades 

&nbsp;                         if bins\[i] <= t\[ 'confidence' ] < bins\[i+ 1 ]]

&nbsp;           

&nbsp;           if  not bin\_trades:

&nbsp;                continue

&nbsp;           

&nbsp;           # Actual accuracy in this bin 

&nbsp;           actual\_accuracy = sum ( 1  for t in bin\_trades if t\[ 'correct' ]) / len (bin\_trades)

&nbsp;           

&nbsp;           # Expected accuracy = mean confidence 

&nbsp;           expected\_accuracy = np.mean(\[t\[ 'confidence' ]/ 100  for t in bin\_trades])

&nbsp;           

&nbsp;           # Calibration error 

&nbsp;           calibration\_error += abs (actual\_accuracy - expected\_accuracy) \* len (bin\_trades)

&nbsp;       

&nbsp;       calibration\_error /= len (trades)

&nbsp;       

&nbsp;       # Perfect calibration = 0, bad = 1 

&nbsp;       return  1.0 - calibration\_error

&nbsp;   

&nbsp;   def \_format\_delta(self, delta: float ) -> str :

&nbsp;        """Format the change""" 

&nbsp;       sign = "+"  if delta >= 0  else  "" 

&nbsp;       direction = "\[UP]"  if delta >= 0  else  "\[DOWN]" 

&nbsp;       return  f" {direction}  {sign} {delta: .2 %} "

Confidence calibration turned out to be a critical metric. I noticed an interesting pattern: as the model degraded, accuracy dropped slowly, but calibration deteriorated rapidly. The model became overconfident in incorrect predictions. SEAL corrected this—after retraining on examples with high confidence and poor performance, the model became more cautious.





Unexpected Discoveries: What a SEAL Learned on His Own

The most surprising thing was what SEAL learned without my intervention. By analyzing high-weight examples in the system's memory, I discovered patterns I'd never programmed myself.



The model learned to recognize false breakouts . It began to associate high quantum entropy + a sharp spike in volume + touching a Bollinger Band with a pullback, even if classic indicators signaled a breakout. I tested this manually—it worked with 73% accuracy.



The second discovery was that the model learned to distinguish between types of volatility . It understood the difference between news-driven volatility (sharp but short-lived) and trend-reversal volatility (gradual but persistent). This insight came from analyzing its own mistakes: trades opened on news-driven volatility often closed at a loss due to a rapid reversal.



Third, the model began to group currency pairs . It understood that EUR/USD and GBP/USD often moved in sync, while USD/CHF moved out of phase. When it saw a strong signal for the euro but a weak one for the pound, this became an additional factor of uncertainty.



All of this arose naturally, from analyzing thousands of transactions. I didn't program these rules—SEAL derived them from experience.





Limitations and problems

SEAL isn't a magic wand. The system has fundamental limitations that need to be understood.



Problem number one: black swans. SEAL learns from experience, meaning it can't predict what it has never seen. The pandemic outbreak in March 2020, the Brexit vote, the depegging of the Swiss franc—in such events, SEAL is useless. In fact, it can be dangerous, because it will confidently predict the continuation of normal market conditions.



Solution: I added an anomaly detector that disables trading during extreme movements.



def is\_market\_abnormal(self, symbol: str ) -> bool :

&nbsp;    """Detects abnormal market conditions"""

&nbsp;   

&nbsp;   df = self.load\_symbol\_data(symbol, bars= 100 )

&nbsp;   

&nbsp;   # Current volatility vs. historical 

&nbsp;   volatility recent\_volatility = df\[ 'close' ].pct\_change().tail( 20 ).std()

&nbsp;   historical\_volatility = df\[ 'close' ].pct\_change().std()

&nbsp;   

&nbsp;   # Anomaly = volatility is 3+ times higher 

&nbsp;   if recent\_volatility > historical\_volatility \* 3 :

&nbsp;       log.warning( f"ANOMALY on {symbol} : volatility {recent\_volatility/historical\_volatility: .1 f} x" )

&nbsp;        return  True

&nbsp;   

&nbsp;   # Checking gaps 

&nbsp;   gaps = abs ((df\[ 'open' ] - df\[ 'close' ].shift( 1 )) / df\[ 'close' ].shift( 1 ))

&nbsp;    if gaps.tail( 5 ) .max () > 0.01 :   # Gap > 1% 

&nbsp;       log.warning( f"ANOMALY on {symbol} : gap {gaps.tail( 5 ).max(): .2 %} " detected )

&nbsp;        return  True

&nbsp;   

&nbsp;   return  False

Problem two: overfitting on success. If the market accidentally enters a mode where a simple strategy works perfectly, SEAL begins to overfit on this success. The model becomes overly aggressive, ignoring risks.



I encountered this in November, when EUR/USD was in a pure uptrend for a whole week. SEAL started opening only long positions, ignoring correction signals. When the trend reversed, the series of losses was painful.



Solution: Added strategy diversity analysis.



def check\_strategy\_diversity(self) -> bool :

&nbsp;    """Check the diversity of trading decisions"""

&nbsp;   

&nbsp;   recent\_trades = self.seal.trade\_memory\[- 50 :]

&nbsp;   

&nbsp;   if  len (recent\_trades) < 30 :

&nbsp;        return  True   # Not enough data

&nbsp;   

&nbsp;   # Calculate the balance of directions 

&nbsp;   up\_trades = sum ( 1  for t in recent\_trades if t\[ 'direction' ] == 'UP' )

&nbsp;   down\_trades = sum ( 1  for t in recent\_trades if t\[ 'direction' ] == 'DOWN' )

&nbsp;   

&nbsp;   balance = min (up\_trades, down\_trades) / max (up\_trades, down\_trades)

&nbsp;   

&nbsp;   if balance < 0.3 :   # More than 70% of trades are in one direction 

&nbsp;       log.warning( f"WARNING: Low diversity of strategies (balance: {balance: .1 %} )" )

&nbsp;        # Raise the confidence threshold for the dominant direction 

&nbsp;       return  False

&nbsp;   

&nbsp;   return  True

Problem three: computational load. Fine-tuning LLM on 200 examples takes 15-20 minutes on an RTX 3090. During periods of high activity, SEAL may run training every 2-3 days. This is normal for a desktop, but problematic for a VPS.



The solution turned out to be quantization and optimization:



\# In Modelfile for finetuning 

PARAMETER num\_gpu 1 

PARAMETER num\_thread 8 

PARAMETER quantization q4\_0   # 4-bit quantization 

PARAMETER batch\_size 4 

PARAMETER epochs 3   # Fewer epochs for faster training

4-bit quantization accelerated training by 2.5 times with minimal loss of quality. Three epochs instead of five saves 40% of the training time. Overall, fine-tuning was reduced.



Here are the results of the system backtest:











A critical look at the backtest results

The backtest showed an almost complete absence of losing trades. This isn't cause for optimism, but a serious warning sign.



In real trading, such results are practically unachievable. They highly likely indicate fundamental problems in the testing or the model itself.



Possible reasons:



Overfitting.

The model is overfitted to historical data and lacks generalization ability. This is a typical error when using complex models on a limited sample.



Look-ahead bias.

Calculations may have implicitly used information unavailable at the time of the trading decision, either directly or through the specifics of feature construction.



Insufficient data.

The small number of trades or short testing period renders the results statistically insignificant and highly unstable.



Ignoring transaction costs.

Spreads, commissions, and slippage can completely destroy apparent profits, especially with high trading frequency.



Parameter overfitting (selection/survivor bias).

If the system parameters were overfitted using the same dataset used to evaluate performance, the backtest loses its diagnostic value.



Currently, these results have not been confirmed by live trading . Until representative statistics are obtained on a real account—at least several hundred trades covering various market conditions—the backtest should be considered purely a preliminary experiment.



Rule of thumb:

If a backtest's results look too good to be true, they almost always are.







The Future of SEAL: Directions for Development

The current version of the SEAL is a basic prototype. Further development is logical in several directions:



Multi-model ensemble.

Instead of a single model, there's a population of 5-7 specialized models. Each is optimized for its own market conditions (trending, flat, high volatility). SEAL selects the active model based on current conditions.



Cross-symbol learning.

Currently, training is performed separately for each instrument. This is a limitation. Correlated pairs (EUR/USD, GBP/USD, etc.) can leverage shared representations and knowledge transfer, accelerating adaptation and reducing overfitting.



Hierarchical memory.

Memory is divided into levels:



short-term - latest transactions and current mode,

medium-term - patterns of weeks and months,

long-term - rare but critically important events.

Each level is used with different frequency and weight in training.

Active learning.

SEAL automatically determines which data is most informative and accelerates learning on complex or rare scenarios, including generating synthetic examples.







Conclusion: Continuous adaptation instead of static models

Working with SEAL demonstrates the limitations of the classic "train → use until degrade" approach. In rapidly changing markets, such systems inevitably lose relevance.



SEAL implements a different principle: continuous learning while preserving context . The model adapts to new conditions without forgetting past patterns, and uses its own trading history as a learning resource.



The key idea is simple:



each transaction is a new learning signal,

every mistake is a model adjustment,

Every successful pattern is confirmed knowledge.

This is not "learning from history", but learning from real results in real time.



Disclaimer:

SEAL is in the experimental stage. Past results do not guarantee future results. Algorithmic trading carries a risk of capital loss and requires independent validation, strict risk management, and a full understanding of the system's limitations.

