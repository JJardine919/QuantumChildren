





# &nbsp;                         ETARE













Introduction

Do you know what evolution, neural networks and traders have in common? They all learn from their mistakes. This is exactly the thought that came to me after yet another sleepless night at the terminal, when my "perfect" trading algorithm once again lost its deposit due to an unexpected market movement.



I remember that day like it was yesterday: June 23, 2016, the Brexit referendum. My algorithm, based on classic technical analysis patterns, confidently held a long position on GBP. "All the polls show that Britain will remain in the EU," I thought then. At 4 a.m. Moscow time, when the first results showed a victory for Brexit supporters, GBP collapsed by 1800 points in a matter of minutes. My deposit lost 40%.



In March 2023, I started developing ETARE - Evolutionary Trading Algorithm with Reinforcement and Extinction (Elimination). Why elimination? Because in nature, the strongest survive. So why not apply this principle to trading strategies?



Are you ready to dive into the world where classic technical analysis meets the latest advances in artificial intelligence? Where every trading strategy struggles for survival in Darwinian natural selection? Then fasten your seat belts – it is going to be interesting. Because what you are about to see is not just another trading robot. It is the result of 15 years of trial and error, thousands of hours of programming and, frankly, a few destroyed deposits. But the main thing is that it is a working system that already brings real profit to its users.





System architecture

At the heart of ETARE is a hybrid architecture reminiscent of a modern quantum computer. Remember the days when we wrote simple scripts for MetaTrader 4 based on the intersection of two moving averages? At the time, it seemed like a breakthrough. Looking back now, I realize we were like ancient sailors trying to cross the ocean using only a compass and the stars.



After the 2022 crash, it became clear that the market is too complex for simple solutions. That is when my journey into the world of machine learning began.



class HybridTrader:

&nbsp;   def \_\_init\_\_(self, symbols, population\_size=50):

&nbsp;       self.population = \[]  # Population of strategies

&nbsp;       self.extinction\_rate = 0.3  # Extinction rate

&nbsp;       self.elite\_size = 5  # Elite individuals

&nbsp;       self.inefficient\_extinction\_interval = 5  # Cleaning interval

Imagine a colony of ants, where each ant is a trading strategy. Strong individuals survive and pass on their genes to their offspring, while weak ones disappear. In my system, the role of genes is played by the weight ratios of the neural network.



Why population\_size=50? Because fewer strategies do not provide sufficient diversification, while more make it difficult to quickly adapt to market changes.



In nature, ants constantly explore new territories, find food and pass on information to their relatives. In ETARE, each strategy also researches the market, and successful trading patterns are passed on to future generations through a cross-breeding mechanism:



def \_crossover(self, parent1, parent2):

&nbsp;   child = TradingIndividual(self.input\_size)

&nbsp;   # Cross scales through a mask

&nbsp;   for attr in \['input\_weights', 'hidden\_weights', 'output\_weights']:

&nbsp;       parent1\_weights = getattr(parent1.weights, attr)

&nbsp;       parent2\_weights = getattr(parent2.weights, attr)

&nbsp;       mask = np.random.random(parent1\_weights.shape) < 0.5

&nbsp;       child\_weights = np.where(mask, parent1\_weights, parent2\_weights)

&nbsp;       setattr(child.weights, attr, child\_weights)

&nbsp;   return child

In December 2024, while analyzing trading logs, I noticed that the most successful codes are often "hybrids" of other successful approaches. Just as in nature, strong genes produce healthy offspring, so in algorithmic trading, successful patterns can combine to create even more efficient strategies.



The heart of the system was the LSTM network, a special type of neural network with "memory". After months of experimenting with various architectures, from simple multilayer perceptrons to complex transformers, we settled on this configuration:



class LSTMModel(nn.Module):

&nbsp;   def \_\_init\_\_(self, input\_size, hidden\_size, output\_size):

&nbsp;       super(LSTMModel, self).\_\_init\_\_()

&nbsp;       self.lstm = nn.LSTM(input\_size, hidden\_size, batch\_first=True)

&nbsp;       self.dropout = nn.Dropout(0.4)  # Protection from overfitting

&nbsp;       self.fc = nn.Linear(hidden\_size, output\_size)

&nbsp;   

&nbsp;   def forward(self, x):

&nbsp;       out, \_ = self.lstm(x)

&nbsp;       out = self.dropout(out\[:, -1, :])  # Use the last LSTM output

&nbsp;       out = self.fc(out)

&nbsp;       return out

Every 100 trades, the system performs "cleaning" ruthlessly removing unprofitable strategies. This is one of the key mechanisms of ETARE, and its creation is a separate story. I remember a night in December 2023 when I was analyzing my trading logs and noticed a surprising pattern: most strategies that had shown losses in the first 100-150 trades continued to be unprofitable thereafter. This observation completely changed the system architecture:



def \_inefficient\_extinction\_event(self):

&nbsp;   """Periodic extinction of inefficient individuals"""

&nbsp;   initial\_size = len(self.population)

&nbsp;   

&nbsp;   # Analyze efficiency of each strategy

&nbsp;   performance\_metrics = \[]

&nbsp;   for individual in self.population:

&nbsp;       metrics = {

&nbsp;           'profit\_factor': individual.total\_profit / abs(individual.max\_drawdown) if individual.max\_drawdown != 0 else 0,

&nbsp;           'win\_rate': len(\[t for t in individual.trade\_history if t.profit > 0]) / len(individual.trade\_history) if individual.trade\_history else 0,

&nbsp;           'risk\_adjusted\_return': individual.total\_profit / individual.volatility if individual.volatility != 0 else 0

&nbsp;       }

&nbsp;       performance\_metrics.append(metrics)

&nbsp;   

&nbsp;   # Remove unprofitable strategies taking into account a comprehensive assessment

&nbsp;   self.population = \[ind for ind, metrics in zip(self.population, performance\_metrics)

&nbsp;                     if metrics\['profit\_factor'] > 1.5 or metrics\['win\_rate'] > 0.6]

&nbsp;   

&nbsp;   # Create new individuals with improved initialization

&nbsp;   while len(self.population) < initial\_size:

&nbsp;       new\_individual = TradingIndividual(self.input\_size)

&nbsp;       new\_individual.mutate()  # Random mutations

&nbsp;       

&nbsp;       # Inherit successful patterns

&nbsp;       if len(self.population) > 0:

&nbsp;           parent = random.choice(self.population)

&nbsp;           new\_individual.inherit\_patterns(parent)

&nbsp;           

&nbsp;       self.population.append(new\_individual)

The trading decision database acts as the system memory. Every decision, every result – everything is recorded for later analysis:



def \_save\_to\_db(self):

&nbsp;   with self.conn:

&nbsp;       self.conn.execute('DELETE FROM population')

&nbsp;       for individual in self.population:

&nbsp;           data = {

&nbsp;               'weights': individual.weights.to\_dict(),

&nbsp;               'fitness': individual.fitness,

&nbsp;               'profit': individual.total\_profit

&nbsp;           }

&nbsp;           self.conn.execute(

&nbsp;               'INSERT INTO population (data) VALUES (?)',

&nbsp;               (json.dumps(data),)

&nbsp;           )

This entire complex mechanism operates as a single organism, constantly evolving and adapting to market changes. During periods of high volatility, such as when the VIX exceeds 25, the system automatically increases the reliability requirements of strategies. And during calm periods, it becomes more aggressive, allowing users to experiment with new trading patterns.







Reinforcement learning mechanism

There is a paradox in developing trading robots: the more complex the algorithm, the worse it performs in the real market. 



That is why we have focused on simplicity and transparency in the ETARE learning mechanism. After two years of experimenting with different architectures, we arrived at a prioritized memory system:



class RLMemory:

&nbsp;   def \_\_init\_\_(self, capacity=10000):

&nbsp;       self.memory = deque(maxlen=capacity)

&nbsp;       self.priorities = deque(maxlen=capacity)

&nbsp;       

&nbsp;   def add(self, state, action, reward, next\_state):

&nbsp;       priority = max(self.priorities) if self.priorities else 1.0

&nbsp;       self.memory.append((state, action, reward, next\_state))

&nbsp;       self.priorities.append(priority)

Every trading decision is more than just a market entry, but a complex balance between risk and potential reward. See how the system learns from its decisions:



def update(self, state, action, reward, next\_state):

&nbsp;   self.memory.add(state, action, reward, next\_state)

&nbsp;   self.total\_profit += reward



&nbsp;   if len(self.memory.memory) >= 32:

&nbsp;       batch = self.memory.sample(32)

&nbsp;       self.\_train\_on\_batch(batch)

I have lost many times because the models could not adapt to a different market situation. It was then that the idea of adaptive learning was born. Now the system analyzes each transaction and adjusts its behavior:



def \_calculate\_confidence(self, prediction, patterns):

&nbsp;   # Baseline confidence from ML model

&nbsp;   base\_confidence = abs(prediction - 0.5) \* 2

&nbsp;   

&nbsp;   # Consider historical experience

&nbsp;   pattern\_confidence = self.\_get\_pattern\_confidence(patterns)

&nbsp;   

&nbsp;   # Dynamic adaptation to the market

&nbsp;   market\_volatility = self.\_get\_current\_volatility()

&nbsp;   return (base\_confidence \* 0.7 + pattern\_confidence \* 0.3) / market\_volatility

The key point is that the system does not just remember successful trades; it learns to understand why they were successful. This is made possible by the multi-layered backpropagation architecture implemented in PyTorch:



def \_train\_on\_batch(self, batch):

&nbsp;   states = torch.FloatTensor(np.array(\[x\[0] for x in batch]))

&nbsp;   actions = torch.LongTensor(np.array(\[x\[1].value for x in batch]))

&nbsp;   rewards = torch.FloatTensor(np.array(\[x\[2] for x in batch]))

&nbsp;   next\_states = torch.FloatTensor(np.array(\[x\[3] for x in batch]))

&nbsp;   

&nbsp;   current\_q = self.forward(states).gather(1, actions.unsqueeze(1))

&nbsp;   next\_q = self.forward(next\_states).max(1)\[0].detach()

&nbsp;   target = rewards + self.gamma \* next\_q

&nbsp;   

&nbsp;   loss = self.criterion(current\_q.squeeze(), target)

&nbsp;   self.optimizer.zero\_grad()

&nbsp;   loss.backward()

&nbsp;   self.optimizer.step()

As a result, we have a system that learns not from ideal backtests, but from real trading experience. Over the past year of live market testing, ETARE has demonstrated its ability to adapt to a variety of market conditions, from calm trends to highly volatile periods.



But the most important thing is that the system continues to evolve. With every trade, with every market loop, it gets a little smarter, a little more efficient. As one of our beta testers said, "This is the first time I have seen an algorithm that actually learns from its mistakes, rather than just adjusting parameters to fit historical data."





The mechanism of extinction of feeble individuals

Charles Darwin never traded in the financial markets, but his theory of evolution provides a remarkable description of the dynamics of successful trading strategies. In nature, it is not the strongest or fastest individuals that survive, but those that best adapt to environmental changes. The same thing happens in the market.



History knows many cases where a "perfect" trading algorithm was obliterated after the first black swan. In 2015, I lost a significant portion of my deposit when the Swiss National Bank unpegged CHF from EUR. My algorithm at that time turned out to be completely unprepared for such an event. This got me thinking: why has nature been able to deal with black swans successfully for millions of years, while our algorithms have not?



The answer came unexpectedly, while reading the book "On the Origin of Species". Darwin described how, during periods of abrupt climate change, it was not the most specialized species that survived, but those that retained the ability to adapt. It is this principle that forms the basis of the extinction mechanism in ETARE:



def \_inefficient\_extinction\_event(self):

&nbsp;   """Periodic extinction of inefficient individuals"""

&nbsp;   initial\_population = len(self.population)

&nbsp;   market\_conditions = self.\_analyze\_market\_state()

&nbsp;   

&nbsp;   # Assessing the adaptability of each strategy

&nbsp;   adaptability\_scores = \[]

&nbsp;   for individual in self.population:

&nbsp;       score = self.\_calculate\_adaptability(

&nbsp;           individual, 

&nbsp;           market\_conditions

&nbsp;       )

&nbsp;       adaptability\_scores.append(score)

&nbsp;   

&nbsp;   # Dynamic survival threshold

&nbsp;   survival\_threshold = np.percentile(

&nbsp;       adaptability\_scores, 

&nbsp;       30  # The bottom 30% of the population is dying out

&nbsp;   )

&nbsp;   

&nbsp;   # Merciless extinction

&nbsp;   survivors = \[]

&nbsp;   for ind, score in zip(self.population, adaptability\_scores):

&nbsp;       if score > survival\_threshold:

&nbsp;           survivors.append(ind)

&nbsp;   

&nbsp;   self.population = survivors

&nbsp;   

&nbsp;   # Restore population through mutations and crossbreeding

&nbsp;   while len(self.population) < initial\_population:

&nbsp;       if len(self.population) >= 2:

&nbsp;           # Crossbreeding of survivors

&nbsp;           parent1 = self.\_tournament\_selection()

&nbsp;           parent2 = self.\_tournament\_selection()

&nbsp;           child = self.\_crossover(parent1, parent2)

&nbsp;       else:

&nbsp;           # Create a new individual 

&nbsp;           child = TradingIndividual(self.input\_size)

&nbsp;       

&nbsp;       # Mutations for adaptation

&nbsp;       child.mutate(market\_conditions.volatility)

&nbsp;       self.population.append(child)

Just as in nature, periods of mass extinction lead to the emergence of new, more advanced species, so in our system, periods of high volatility become a catalyst for the evolution of strategies. Take a look at the mechanism of natural selection:



def \_extinction\_event(self):

&nbsp;   # Analyze market conditions

&nbsp;   market\_phase = self.\_identify\_market\_phase()

&nbsp;   volatility = self.\_calculate\_market\_volatility()

&nbsp;   trend\_strength = self.\_measure\_trend\_strength()

&nbsp;   

&nbsp;   # Adaptive sorting by survival

&nbsp;   def fitness\_score(individual):

&nbsp;       return (

&nbsp;           individual.profit\_factor \* 0.4 +

&nbsp;           individual.sharp\_ratio \* 0.3 +

&nbsp;           individual.adaptability\_score \* 0.3

&nbsp;       ) \* (1 + individual.correlation\_with\_market)

&nbsp;   

&nbsp;   self.population.sort(

&nbsp;       key=fitness\_score, 

&nbsp;       reverse=True

&nbsp;   )

&nbsp;   

&nbsp;   # Preserve elite with diversity in mind

&nbsp;   elite\_size = max(

&nbsp;       5, 

&nbsp;       int(len(self.population) \* 0.1)

&nbsp;   )

&nbsp;   survivors = self.population\[:elite\_size]

&nbsp;   

&nbsp;   # Create a new generation

&nbsp;   while len(survivors) < self.population\_size:

&nbsp;       if random.random() < 0.8:  # 80% crossover

&nbsp;           # Tournament selection of parents

&nbsp;           parent1 = self.\_tournament\_selection()

&nbsp;           parent2 = self.\_tournament\_selection()

&nbsp;           

&nbsp;           # Crossbreeding considering account market conditions

&nbsp;           child = self.\_adaptive\_crossover(

&nbsp;               parent1, 

&nbsp;               parent2, 

&nbsp;               market\_phase

&nbsp;           )

&nbsp;       else:  # 20% elite mutation

&nbsp;           # Clone with mutations

&nbsp;           template = random.choice(survivors\[:3])

&nbsp;           child = self.\_clone\_with\_mutations(

&nbsp;               template,

&nbsp;               volatility,

&nbsp;               trend\_strength

&nbsp;           )

&nbsp;       survivors.append(child)

We paid special attention to the fitness assessment mechanism. In nature, this is the ability of an individual to produce viable offspring; in our case, it is the ability of a strategy to generate profit in various market conditions:



def evaluate\_fitness(self, individual):

&nbsp;   # Basic metrics

&nbsp;   profit\_factor = individual.total\_profit / max(

&nbsp;       abs(individual.total\_loss), 

&nbsp;       1e-6

&nbsp;   )

&nbsp;   

&nbsp;   # Resistance to drawdowns

&nbsp;   max\_dd = max(individual.drawdown\_history) if individual.drawdown\_history else 0

&nbsp;   drawdown\_resistance = 1 / (1 + max\_dd)

&nbsp;   

&nbsp;   # Profit sequence analysis

&nbsp;   profit\_sequence = \[t.profit for t in individual.trade\_history\[-50:]]

&nbsp;   consistency = self.\_analyze\_profit\_sequence(profit\_sequence)

&nbsp;   

&nbsp;   # Correlation with the market

&nbsp;   market\_correlation = self.\_calculate\_market\_correlation(

&nbsp;       individual.trade\_history

&nbsp;   )

&nbsp;   

&nbsp;   # Adaptability to changes

&nbsp;   adaptability = self.\_measure\_adaptability(

&nbsp;       individual.performance\_history

&nbsp;   )

&nbsp;   

&nbsp;   # Comprehensive assessment

&nbsp;   fitness = (

&nbsp;       profit\_factor \* 0.3 +

&nbsp;       drawdown\_resistance \* 0.2 +

&nbsp;       consistency \* 0.2 +

&nbsp;       (1 - abs(market\_correlation)) \* 0.1 +

&nbsp;       adaptability \* 0.2

&nbsp;   )

&nbsp;   

&nbsp;   return fitness

This is how the mutation of surviving strategies occurs. This process is reminiscent of genetic mutations in nature, where random changes in DNA sometimes lead to the emergence of more viable organisms:



def mutate(self, market\_conditions):

&nbsp;   """Adaptive mutation considering market conditions"""

&nbsp;   # Dynamic adjustment of mutation strength

&nbsp;   self.mutation\_strength = self.\_calculate\_mutation\_strength(

&nbsp;       market\_conditions.volatility,

&nbsp;       market\_conditions.trend\_strength

&nbsp;   )

&nbsp;   

&nbsp;   if np.random.random() < self.mutation\_rate:

&nbsp;       # Mutation of neural network weights

&nbsp;       for weight\_matrix in \[

&nbsp;           self.weights.input\_weights,

&nbsp;           self.weights.hidden\_weights,

&nbsp;           self.weights.output\_weights

&nbsp;       ]:

&nbsp;           # Mutation mask with adaptive threshold

&nbsp;           mutation\_threshold = 0.1 \* (

&nbsp;               1 + market\_conditions.uncertainty

&nbsp;           )

&nbsp;           mask = np.random.random(weight\_matrix.shape) < mutation\_threshold

&nbsp;           

&nbsp;           # Volatility-aware mutation generation

&nbsp;           mutations = np.random.normal(

&nbsp;               0,

&nbsp;               self.mutation\_strength \* market\_conditions.volatility,

&nbsp;               size=mask.sum()

&nbsp;           )

&nbsp;           

&nbsp;           # Apply mutations

&nbsp;           weight\_matrix\[mask] += mutations

&nbsp;           

&nbsp;       # Mutation of hyperparameters

&nbsp;       if random.random() < 0.3:  # 30% chance

&nbsp;           self.\_mutate\_hyperparameters(market\_conditions)

Interestingly, in some versions of the system, during periods of high market volatility, the system automatically increases the intensity of mutations. This is reminiscent of how some bacteria accelerate mutations under stressful conditions. In our case:



def \_calculate\_mutation\_strength(self, volatility, trend\_strength):

&nbsp;   """Calculate mutation strength based on market conditions"""

&nbsp;   base\_strength = self.base\_mutation\_strength

&nbsp;   

&nbsp;   # Mutation enhancement under high volatility

&nbsp;   volatility\_factor = 1 + (volatility / self.average\_volatility - 1)

&nbsp;   

&nbsp;   # Weaken mutations in a strong trend

&nbsp;   trend\_factor = 1 / (1 + trend\_strength)

&nbsp;   

&nbsp;   # Mutation total strength

&nbsp;   mutation\_strength = (

&nbsp;       base\_strength \* 

&nbsp;       volatility\_factor \* 

&nbsp;       trend\_factor

&nbsp;   )

&nbsp;   

&nbsp;   return np.clip(

&nbsp;       mutation\_strength,

&nbsp;       self.min\_mutation\_strength,

&nbsp;       self.max\_mutation\_strength

&nbsp;   )

The mechanism of population diversification is especially important. In nature, genetic diversity is the key to species survival. In ETARE, we have implemented a similar principle:



def \_maintain\_population\_diversity(self):

&nbsp;   """ Maintain diversity in the population"""

&nbsp;   # Calculate the strategy similarity matrix

&nbsp;   similarity\_matrix = np.zeros(

&nbsp;       (len(self.population), len(self.population))

&nbsp;   )

&nbsp;   

&nbsp;   for i, ind1 in enumerate(self.population):

&nbsp;       for j, ind2 in enumerate(self.population\[i+1:], i+1):

&nbsp;           similarity = self.\_calculate\_strategy\_similarity(ind1, ind2)

&nbsp;           similarity\_matrix\[i,j] = similarity\_matrix\[j,i] = similarity

&nbsp;   

&nbsp;   # Identify clusters of similar strategies

&nbsp;   clusters = self.\_identify\_strategy\_clusters(similarity\_matrix)

&nbsp;   

&nbsp;   # Forced diversification when necessary

&nbsp;   for cluster in clusters:

&nbsp;       if len(cluster) > self.max\_cluster\_size:

&nbsp;           # We leave only the best strategies in the cluster

&nbsp;           survivors = sorted(

&nbsp;               cluster,

&nbsp;               key=lambda x: x.fitness,

&nbsp;               reverse=True

&nbsp;           )\[:self.max\_cluster\_size]

&nbsp;           

&nbsp;           # Replace the rest with new strategies

&nbsp;           for idx in cluster\[self.max\_cluster\_size:]:

&nbsp;               self.population\[idx] = TradingIndividual(

&nbsp;                   self.input\_size,

&nbsp;                   mutation\_rate=self.high\_mutation\_rate

&nbsp;               )

Result? The system that does not just trade, but evolves with the market. As Darwin said, it is not the strongest that survives, but the most adaptive. In the world of algorithmic trading, this is more relevant than ever.





Trading decisions database

Maintaining trading experience is just as important as gaining it. Over the years of working with algorithmic systems, I have repeatedly become convinced that without a reliable database, any trading system will sooner or later "forget" its best strategies. In ETARE, we have implemented multi-level storage for trading decisions:



def \_create\_tables(self):

&nbsp;   """ Create a database structure"""

&nbsp;   with self.conn:

&nbsp;       self.conn.execute('''

&nbsp;           CREATE TABLE IF NOT EXISTS population (

&nbsp;               id INTEGER PRIMARY KEY,

&nbsp;               individual TEXT,

&nbsp;               created\_at TIMESTAMP DEFAULT CURRENT\_TIMESTAMP,

&nbsp;               last\_update TIMESTAMP

&nbsp;           )

&nbsp;       ''')

&nbsp;       

&nbsp;       self.conn.execute('''

&nbsp;           CREATE TABLE IF NOT EXISTS history (

&nbsp;               id INTEGER PRIMARY KEY,

&nbsp;               generation INTEGER,

&nbsp;               individual\_id INTEGER,

&nbsp;               trade\_history TEXT,

&nbsp;               market\_conditions TEXT,

&nbsp;               FOREIGN KEY(individual\_id) REFERENCES population(id)

&nbsp;           )

&nbsp;       ''')

Every trade, every decision, even those that seem insignificant, become part of the system collective experience. Here is how we save data after each trading loop:



def \_save\_to\_db(self):

&nbsp;   try:

&nbsp;       with self.conn:

&nbsp;           self.conn.execute('DELETE FROM population')

&nbsp;           for individual in self.population:

&nbsp;               individual\_data = {

&nbsp;                   'weights': {

&nbsp;                       'input\_weights': individual.weights.input\_weights.tolist(),

&nbsp;                       'hidden\_weights': individual.weights.hidden\_weights.tolist(),

&nbsp;                       'output\_weights': individual.weights.output\_weights.tolist(),

&nbsp;                       'hidden\_bias': individual.weights.hidden\_bias.tolist(),

&nbsp;                       'output\_bias': individual.weights.output\_bias.tolist()

&nbsp;                   },

&nbsp;                   'fitness': individual.fitness,

&nbsp;                   'total\_profit': individual.total\_profit,

&nbsp;                   'trade\_history': list(individual.trade\_history),

&nbsp;                   'market\_metadata': self.\_get\_market\_conditions()

&nbsp;               }

&nbsp;               self.conn.execute(

&nbsp;                   'INSERT INTO population (individual) VALUES (?)', 

&nbsp;                   (json.dumps(individual\_data),)

&nbsp;               )

&nbsp;   except Exception as e:

&nbsp;       logging.error(f"Error saving population: {str(e)}")

Even after a critical server failure, the entire system will be restored in just minutes, thanks to detailed logs and backups. Here is how the recovery mechanism works:



def \_load\_from\_db(self):

&nbsp;   """Load population from database"""

&nbsp;   try:

&nbsp;       cursor = self.conn.execute('SELECT individual FROM population')

&nbsp;       rows = cursor.fetchall()

&nbsp;       for row in rows:

&nbsp;           individual\_data = json.loads(row\[0])

&nbsp;           individual = TradingIndividual(self.input\_size)

&nbsp;           individual.weights = GeneticWeights(\*\*individual\_data\['weights'])

&nbsp;           individual.fitness = individual\_data\['fitness']

&nbsp;           individual.total\_profit = individual\_data\['total\_profit']

&nbsp;           individual.trade\_history = deque(

&nbsp;               individual\_data\['trade\_history'], 

&nbsp;               maxlen=1000

&nbsp;           )

&nbsp;           self.population.append(individual)

&nbsp;   except Exception as e:

&nbsp;       logging.error(f"Error loading population: {str(e)}")

We will pay special attention to the analysis of historical data. Every successful strategy leaves a trace that can be used to improve future decisions:



def analyze\_historical\_performance(self):

&nbsp;   """ Historical performance analysis"""

&nbsp;   query = '''

&nbsp;       SELECT h.\*, p.individual 

&nbsp;       FROM history h 

&nbsp;       JOIN population p ON h.individual\_id = p.id 

&nbsp;       WHERE h.generation > ? 

&nbsp;       ORDER BY h.generation DESC

&nbsp;   '''

&nbsp;   

&nbsp;   cursor = self.conn.execute(query, (self.generation - 100,))

&nbsp;   performance\_data = cursor.fetchall()

&nbsp;   

&nbsp;   # Analyze patterns of successful strategies

&nbsp;   success\_patterns = defaultdict(list)

&nbsp;   for record in performance\_data:

&nbsp;       trade\_data = json.loads(record\[3])

&nbsp;       if trade\_data\['profit'] > 0:

&nbsp;           market\_conditions = json.loads(record\[4])

&nbsp;           key\_pattern = self.\_extract\_key\_pattern(market\_conditions)

&nbsp;           success\_patterns\[key\_pattern].append(trade\_data)

&nbsp;   

&nbsp;   return success\_patterns

The ETARE database is not just a storage facility for information, but the true "brain" of the system, capable of analyzing the past and predicting the future. As my old mentor used to say: "A trading system without memory is like a trader without experience: he starts from scratch every day".





Data and features

Over the years of working with algorithmic trading, I have tried hundreds of indicator combinations. At one point, my trading system used more than 50 different indicators, from the classic RSI to exotic indicators of my own design. But do you know what I realized after another lost deposit? It is not about quantity, but about proper data handling.



I remember an incident during Brexit: a system with dozens of indicators simply "froze" being unable to make a decision due to conflicting signals. That is when the idea for ETARE was born – a system that uses the minimum necessary set of indicators, but handles them in an intelligent way.



def prepare\_features(data: pd.DataFrame) -> pd.DataFrame:

&nbsp;   """Prepare features for analysis"""

&nbsp;   df = data.copy()



&nbsp;   # RSI - as an overbought/oversold detector

&nbsp;   delta = df\['close'].diff()

&nbsp;   gain = delta.where(delta > 0, 0).rolling(14).mean()

&nbsp;   loss = -delta.where(delta < 0, 0).rolling(14).mean()

&nbsp;   rs = gain / loss

&nbsp;   df\['rsi'] = 100 - (100 / (1 + rs))

RSI in our system is not just an overbought/oversold indicator. We use it as part of a comprehensive analysis of market sentiment. It works especially effectively in combination with MACD:

\# MACD - to determine the trend

&nbsp;   exp1 = df\['close'].ewm(span=12, adjust=False).mean()

&nbsp;   exp2 = df\['close'].ewm(span=26, adjust=False).mean()

&nbsp;   df\['macd'] = exp1 - exp2

&nbsp;   df\['macd\_signal'] = df\['macd'].ewm(span=9, adjust=False).mean()

&nbsp;   df\['macd\_hist'] = df\['macd'] - df\['macd\_signal']

Bollinger Bands are our volatility "radar".  



\# Bollinger Bands with adaptive period

&nbsp;   volatility = df\['close'].rolling(50).std()

&nbsp;   adaptive\_period = int(20 \* (1 + volatility.mean()))

&nbsp;   

&nbsp;   df\['bb\_middle'] = df\['close'].rolling(adaptive\_period).mean()

&nbsp;   df\['bb\_std'] = df\['close'].rolling(adaptive\_period).std()

&nbsp;   df\['bb\_upper'] = df\['bb\_middle'] + 2 \* df\['bb\_std']

&nbsp;   df\['bb\_lower'] = df\['bb\_middle'] - 2 \* df\['bb\_std']

A separate story is the analysis of volatility and momentum. 



\# Momentum - market "temperature"

&nbsp;   df\['momentum'] = df\['close'] / df\['close'].shift(10)

&nbsp;   df\['momentum\_ma'] = df\['momentum'].rolling(20).mean()

&nbsp;   df\['momentum\_std'] = df\['momentum'].rolling(20).std()

&nbsp;   

&nbsp;   # Volatility is our "seismograph"

&nbsp;   df\['atr'] = df\['high'].rolling(14).max() - df\['low'].rolling(14).min()

&nbsp;   df\['price\_change'] = df\['close'].pct\_change()

&nbsp;   df\['price\_change\_abs'] = df\['price\_change'].abs()

&nbsp;   

&nbsp;   # Volume volatility

&nbsp;   df\['volume\_volatility'] = df\['tick\_volume'].rolling(20).std() / df\['tick\_volume'].rolling(20).mean()

Volume analysis in ETARE is more than just tick counting. We have developed a dedicated algorithm for detecting abnormal volumes that helps predict strong movements:



\# Volume analysis - market "pulse"

&nbsp;   df\['volume\_ma'] = df\['tick\_volume'].rolling(20).mean()

&nbsp;   df\['volume\_std'] = df\['tick\_volume'].rolling(20).std()

&nbsp;   df\['volume\_ratio'] = df\['tick\_volume'] / df\['volume\_ma']

&nbsp;   

&nbsp;   # Detection of abnormal volumes

&nbsp;   df\['volume\_spike'] = (

&nbsp;       df\['tick\_volume'] > df\['volume\_ma'] + 2 \* df\['volume\_std']

&nbsp;   ).astype(int)

&nbsp;   

&nbsp;   # Cluster analysis of volumes

&nbsp;   df\['volume\_cluster'] = (

&nbsp;       df\['tick\_volume'].rolling(3).sum() / 

&nbsp;       df\['tick\_volume'].rolling(20).sum()

&nbsp;   )

The final touch is data normalization. This is a critical step that many people underestimate.



\# Normalization considering market phases

&nbsp;   numeric\_cols = df.select\_dtypes(include=\[np.number]).columns

&nbsp;   for col in numeric\_cols:

&nbsp;       # Adaptive normalization

&nbsp;       rolling\_mean = df\[col].rolling(100).mean()

&nbsp;       rolling\_std = df\[col].rolling(100).std()

&nbsp;       df\[col] = (df\[col] - rolling\_mean) / (rolling\_std + 1e-8)

&nbsp;   

&nbsp;   # Removing outliers

&nbsp;   df = df.clip(-4, 4)  # Limit values to the range \[-4, 4]

&nbsp;   

&nbsp;   return df

Each indicator in ETARE is not just a number, but part of a complex mosaic of market analysis. The system constantly adapts to market changes, adjusting the weight of each indicator depending on the current situation. In the following sections, we will see how this data is translated into actual trading decisions.





Trading Logic

I present to you a description of an innovative trading system that embodies cutting-edge algorithmic trading technologies. The system is based on a hybrid approach that combines genetic optimization, machine learning, and advanced risk management.



The heart of the system is a continuously operating trading loop that constantly analyzes market conditions and adapts to them. Like natural evolution, the system periodically "cleans" ineffective trading strategies, giving way to new, more promising approaches. This happens every 50 trades, ensuring continuous improvement of trading algorithms.



Each trading instrument is handled individually, taking into account its unique characteristics. The system analyzes historical data for the last 100 candles, which allows it to form an accurate picture of the current market state. Based on this analysis, informed decisions are made about opening and closing positions.



Particular attention is paid to the position averaging strategy (DCA). When opening new positions, the system automatically reduces their volume, starting from 0.1 lot and gradually decreasing to the minimum value of 0.01 lot. This allows for efficient management of risks and maximization of potential profits.



The process of closing positions is also carefully thought out. The system monitors the profitability of each position and closes them when a specified profit level is reached. In this case, Buy and Sell positions are handled separately, which allows for more flexible portfolio management. The rewards or penalties received as a result of trading are the key to further successful learning. 



All information about trading operations and system status is stored in the database, providing the ability to perform detailed analysis and optimize strategies. This creates a solid foundation for further improvement of trading algorithms.



&nbsp;   def \_process\_individual(self, symbol: str, individual: TradingIndividual, current\_state: np.ndarray):

&nbsp;       """Handle trading logic for an individual using DCA and split closing by profit"""

&nbsp;       try:

&nbsp;           positions = individual.open\_positions.get(symbol, \[])



&nbsp;           if not positions:  # Open a new position

&nbsp;               action, \_ = individual.predict(current\_state)

&nbsp;               if action in \[Action.OPEN\_BUY, Action.OPEN\_SELL]:

&nbsp;                   self.\_open\_position(symbol, individual, action)

&nbsp;           else:  # Manage existing positions

&nbsp;               current\_price = mt5.symbol\_info\_tick(symbol).bid



&nbsp;               # Close positions by profit

&nbsp;               self.\_close\_positions\_by\_profit(symbol, individual, current\_price)



&nbsp;               # Check for the need to open a new position by DCA

&nbsp;               if len(positions) < self.max\_positions\_per\_pair:

&nbsp;                   action, \_ = individual.predict(current\_state)

&nbsp;                   if action in \[Action.OPEN\_BUY, Action.OPEN\_SELL]:

&nbsp;                       self.\_open\_dca\_position(symbol, individual, action, len(positions))



&nbsp;       except Exception as e:

&nbsp;           logging.error(f"Error processing individual: {str(e)}")



&nbsp;   def \_open\_position(self, symbol: str, individual: TradingIndividual, action: Action):

&nbsp;       """Open a position"""

&nbsp;       try:

&nbsp;           volume = 0.1

&nbsp;           price = mt5.symbol\_info\_tick(symbol).ask if action == Action.OPEN\_BUY else mt5.symbol\_info\_tick(symbol).bid



&nbsp;           request = {

&nbsp;               "action": mt5.TRADE\_ACTION\_DEAL,

&nbsp;               "symbol": symbol,

&nbsp;               "volume": volume,

&nbsp;               "type": mt5.ORDER\_TYPE\_BUY if action == Action.OPEN\_BUY else mt5.ORDER\_TYPE\_SELL,

&nbsp;               "price": price,

&nbsp;               "deviation": 20,

&nbsp;               "magic": 123456,

&nbsp;               "comment": f"Gen{self.generation}",

&nbsp;               "type\_time": mt5.ORDER\_TIME\_GTC,

&nbsp;               "type\_filling": mt5.ORDER\_FILLING\_FOK,

&nbsp;           }



&nbsp;           result = mt5.order\_send(request)

&nbsp;           if result and result.retcode == mt5.TRADE\_RETCODE\_DONE:

&nbsp;               trade = Trade(symbol=symbol, action=action, volume=volume,

&nbsp;                             entry\_price=result.price, entry\_time=time.time())

&nbsp;               if symbol not in individual.open\_positions:

&nbsp;                   individual.open\_positions\[symbol] = \[]

&nbsp;               individual.open\_positions\[symbol].append(trade)



&nbsp;       except Exception as e:

&nbsp;           logging.error(f"Error opening position: {str(e)}")



&nbsp;   def \_open\_dca\_position(self, symbol: str, individual: TradingIndividual, action: Action, position\_count: int):

&nbsp;       """Open a position using the DCA strategy"""

&nbsp;       try:

&nbsp;           # Basic volume

&nbsp;           base\_volume = 0.1  # Initial volume in lots

&nbsp;           # Reduce the volume by 0.01 lot for each subsequent position

&nbsp;           volume = max(0.01, base\_volume - (position\_count \* 0.01))  # Minimum volume of 0.01 lots

&nbsp;           price = mt5.symbol\_info\_tick(symbol).ask if action == Action.OPEN\_BUY else mt5.symbol\_info\_tick(symbol).bid



&nbsp;           request = {

&nbsp;               "action": mt5.TRADE\_ACTION\_DEAL,

&nbsp;               "symbol": symbol,

&nbsp;               "volume": volume,

&nbsp;               "type": mt5.ORDER\_TYPE\_BUY if action == Action.OPEN\_BUY else mt5.ORDER\_TYPE\_SELL,

&nbsp;               "price": price,

&nbsp;               "deviation": 20,

&nbsp;               "magic": 123456,

&nbsp;               "comment": f"Gen{self.generation} DCA",

&nbsp;               "type\_time": mt5.ORDER\_TIME\_GTC,

&nbsp;               "type\_filling": mt5.ORDER\_FILLING\_FOK,

&nbsp;           }



&nbsp;           result = mt5.order\_send(request)

&nbsp;           if result and result.retcode == mt5.TRADE\_RETCODE\_DONE:

&nbsp;               trade = Trade(symbol=symbol, action=action, volume=volume,

&nbsp;                             entry\_price=result.price, entry\_time=time.time())

&nbsp;               if symbol not in individual.open\_positions:

&nbsp;                   individual.open\_positions\[symbol] = \[]

&nbsp;               individual.open\_positions\[symbol].append(trade)



&nbsp;       except Exception as e:

&nbsp;           logging.error(f"Error opening DCA position: {str(e)}")



&nbsp;   def \_close\_positions\_by\_profit(self, symbol: str, individual: TradingIndividual, current\_price: float):

&nbsp;       """Close positions by profit separately for Buy and Sell"""

&nbsp;       try:

&nbsp;           positions = individual.open\_positions.get(symbol, \[])

&nbsp;           buy\_positions = \[pos for pos in positions if pos.action == Action.OPEN\_BUY]

&nbsp;           sell\_positions = \[pos for pos in positions if pos.action == Action.OPEN\_SELL]



&nbsp;           # Close Buy positions

&nbsp;           for position in buy\_positions:

&nbsp;               profit = calculate\_profit(position, current\_price)

&nbsp;               if profit >= self.min\_profit\_pips:

&nbsp;                   self.\_close\_position(symbol, individual, position)



&nbsp;           # Close Sell positions

&nbsp;           for position in sell\_positions:

&nbsp;               profit = calculate\_profit(position, current\_price)

&nbsp;               if profit >= self.min\_profit\_pips:

&nbsp;                   self.\_close\_position(symbol, individual, position)



&nbsp;       except Exception as e:

&nbsp;           logging.error(f"Error closing positions by profit: {str(e)}")



&nbsp;   def \_close\_position(self, symbol: str, individual: TradingIndividual, position: Trade):

&nbsp;       """Close a position with a model update"""

&nbsp;       try:

&nbsp;           close\_type = mt5.ORDER\_TYPE\_SELL if position.action == Action.OPEN\_BUY else mt5.ORDER\_TYPE\_BUY

&nbsp;           price = mt5.symbol\_info\_tick(symbol).bid if close\_type == mt5.ORDER\_TYPE\_SELL else mt5.symbol\_info\_tick(symbol).ask



&nbsp;           request = {

&nbsp;               "action": mt5.TRADE\_ACTION\_DEAL,

&nbsp;               "symbol": symbol,

&nbsp;               "volume": position.volume,

&nbsp;               "type": close\_type,

&nbsp;               "price": price,

&nbsp;               "deviation": 20,

&nbsp;               "magic": 123456,

&nbsp;               "comment": "Close",

&nbsp;               "type\_time": mt5.ORDER\_TIME\_GTC,

&nbsp;               "type\_filling": mt5.ORDER\_FILLING\_FOK,

&nbsp;           }



&nbsp;           result = mt5.order\_send(request)

&nbsp;           if result and result.retcode == mt5.TRADE\_RETCODE\_DONE:

&nbsp;               position.is\_open = False

&nbsp;               position.exit\_price = result.price

&nbsp;               position.exit\_time = time.time()

&nbsp;               position.profit = calculate\_profit(position, result.price)

&nbsp;               

&nbsp;               # Generate data for training

&nbsp;               trade\_data = {

&nbsp;                   'symbol': symbol,

&nbsp;                   'action': position.action,

&nbsp;                   'entry\_price': position.entry\_price,

&nbsp;                   'exit\_price': position.exit\_price,

&nbsp;                   'volume': position.volume,

&nbsp;                   'profit': position.profit,

&nbsp;                   'holding\_time': position.exit\_time - position.entry\_time

&nbsp;               }

&nbsp;               

&nbsp;               # Update the model with new data

&nbsp;               individual.model.update(trade\_data)

&nbsp;               

&nbsp;               # Save history and update open positions

&nbsp;               individual.trade\_history.append(position)

&nbsp;               individual.open\_positions\[symbol].remove(position)

&nbsp;               

&nbsp;               # Log training results

&nbsp;               logging.info(f"Model updated with trade data: {trade\_data}")



&nbsp;       except Exception as e:

&nbsp;           logging.error(f"Error closing position: {str(e)}")



def main():

&nbsp;   symbols = \['EURUSD.ecn', 'GBPUSD.ecn', 'USDJPY.ecn', 'AUDUSD.ecn']

&nbsp;   trader = HybridTrader(symbols)

&nbsp;   trader.run\_trading\_cycle()



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   main()



The result is a reliable, self-learning trading system capable of operating efficiently in a variety of market conditions. The combination of evolutionary algorithms, machine learning, and proven trading strategies makes it a powerful tool for modern trading.





Conclusion

In conclusion, I would like to emphasize that ETARE is not just another trading algorithm, but the result of many years of evolution in algorithmic trading. The system combines best practices from various fields: genetic algorithms for adaptation to changing market conditions, deep learning for decision-making, and classical risk management methods.



ETARE's uniqueness lies in its ability to continuously learn from its own experiences. Every trade, regardless of outcome, becomes part of the system's collective memory, helping to improve future trading decisions. The mechanism of natural selection of trading strategies, inspired by Darwin's theory of evolution, ensures the survival of only the most effective approaches.



During development and testing, the system has proven its resilience in a variety of market conditions, from calm trend movements to highly volatile periods. It is especially important to note the efficiency of the DCA strategy and the mechanism of separate position closing, which allow us to maximize profits while controlling the level of risk.



Now, regarding efficiency. I will say it straight out: the main ETARE module itself does not trade for me. It is integrated, as a module, into the wider Midas trading ecosystem.







There are currently 24 modules in Midas, including this one. The complexity will increase steadily, and I will describe a lot of it in future articles. 







The future of algorithmic trading lies precisely in such adaptive systems, capable of evolving with the market. ETARE is a step in this direction, demonstrating how modern technologies can be applied to create reliable and profitable trading solutions.

