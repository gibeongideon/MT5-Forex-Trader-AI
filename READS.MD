MQL5 Wizard Techniques you should know (Part 93): Using Suffix Automation and an Auto Encoder in a Custom Money Management Class
MetaTrader 5 — Trading systems | today, 15:46
219
0
Stephen Njuki
Stephen Njuki
Introduction
The problem with traditional money management is not just that it counts lots, but that it reacts too late—it relies on losses that have already occurred. For geometric or pattern traders, this lagging approach is especially dangerous. In this article, we build `CMoneySuffixAE`, a custom risk model that differs fundamentally from standard built-in classes. Instead of looking at the history of closed losses, it proactively sizes positions based on current market structure and real-time geometric integrity.

We resume this series on the MQL5 Wizard where we last considered an implementation of a custom entry signal based on the Fenwick Tree and a CNN. In principle, the MQL5 Wizard supports three customization modes: Trailing Stop, Entry Signal, and Money Management. This series shows how to customize all three modes for different trading styles.

Our Last Outing: volume-based money management
In this article, we return to a custom Money Management implementation. The last time we implemented such a custom class, we used the Fenwick Tree (Binary Indexed Tree) to monitor Volume as measured by the On-Balance-Volume indicator and paired this algorithm with a 1D Convolutional Neural network (CNN) for feature smoothing. Position-sizing strategies are not one-size-fits-all; they vary by trading philosophy and market regime. In the last article on custom money management, we had an implementation that was tailor-made for momentum traders who like exploiting trend-heavy situations. This was because we were proactively sizing trades based on continuous accumulation volume. However, what if some traders' strategies/outlook do not pay attention to volume? For many forex traders this is actually not an anomaly.

Case for Geometric Risk Model
There are traders whose focus is on algorithmic patterns or navigating highly structured geometric markets. And what are geometric markets? Well, these could be understood as markets at fractal points or highly recognizable, and historically repeating patterns. Examples of these often include: harmonic patterns (think Gartley, Bat, Butterfly); Elliot Wave Sequences; Typical Chart-Patterns (e.g. Head & Shoulders, Triangles, Wedges etc.). These are readable sequential markets that could be range-bound or trending. 

For a geometric trader, therefore, trade volume is secondary versus historical price action. Arguably, these traders tend to proactively scale down position sizes based on repeated recent price patterns. This would be done while concurrently shielding capital from sudden unprecedented structural anomalies. Purely reactive, performance based risk models often are poorly equipped to handle these fractal-type of situations. 

Our Problem: Structural Breaks
To construct a robust Expert Advisor, one needs to curate the math tools of his respective market environment. A momentum trader depends on volume accumulation as we saw in the last implementation of a custom money management class with the Fenwick Tree. However, for a geometric trader - who hunts for precise price action patterns or fractals - volume would be on the back-burner. Their strategy's lifeblood is historical repetition.

Nonetheless, geometric trading does carry a special fatal vulnerability: The 'structural break'. This often happens when there is a macroeconomic shock that seriously severs the market's expected geometric rhythm. As a result, previously reliable patterns can fail spectacularly. If these traders were using built-in classes such as the 'CMoneySizeOptimized' they would be unprepared to handle these anomalies. 

If we examine the code of this built-in class we can see it depends on a lagging loss-counter. Its policy is to reduce position size after a string of trades are closed as losses. If a trader relies on geometric patterns, then such a whipsaw environment where major news is breaking would leave him with a strategy that is reactive to loss-counters. If the market breaks structure, waiting to absorb three losses before reducing risk can cripple a portfolio.

Model idea: two engines instead of one loss counter
We therefore set out to build a specialized, proactive engine. This necessitates two components: a sequencing algorithm that is able to map accurate price patterns in real-time so as to confidently scale up on volume once setups are validated; and secondly an anomaly detector that aggressively slashes risk as soon as the market's geometry becomes unreadable - thus preserving capital way before the first loss is realized. 

Here's a short summary: the Suffix Automaton looks for pattern familiarity, while the Autoencoder checks the integrity of the structure.



Model Description
To clarify the potential of this geometry-based model, we step away from MQL5 syntax and focus on the underlying algorithmic and network concepts. This comparison highlights how the two pairs fit different trading environments. Replacing the Fenwick Tree with a Suffix Automaton is like swapping an accountant for a geneticist. Replacing the 1D CNN with an autoencoder is like replacing a camera filter with a compression artist.

Fenwick Tree vs Suffix Automaton

In the last custom money management article, the Fenwick Tree served as a meticulous accountant. An accountant keeps tabs on cumulative totals - that in our case were the continuous running sum of bullish or bearish volume getting into the market. It was able to excel at addressing the question, 'How much total pressure is driving this move?' Because of this, it was the suitable tool for momentum traders riding heaving trends, but an Accountant on the other hand would be blind to the particular shape of the market.

The geometric trader though, unlike momentum followers, does not dwell on total volume but cares about precise historical repetition. Therefore, if we replace the Accountant with a Geneticist, he would ignore the volume and instead translate erratic chart raw prices into a clean discrete 'Price DNA' alphabet. To illustrate simply, a higher close is marked 'Up' (U); while a lower close would be 'Down' (D) and no change could be 'Flat' (F).

When our Expert Advisor proposes a trade, it hands the Geneticist the recent, say, 16-bar DNA sequence. Given that the Suffix Automaton has already mapped the whole market history into a highly tuned index, the Geneticist would not need to go through the history book page by page. The read off is immediate in the form: 'This precise geometric pattern has a 90% historical match rate.' This approach allows the pattern trader to scale up or down their lot size based on exact sequence familiarity as opposed to cumulative momentum.

1D CNN vs Autoencoder

Also, in the last article with the custom money management class our neural network behaved like a specialized Camera Filter. This was the 1D CNN. We were sliding a math window across continuous data, to smooth out erratic noise in order to highlight the actual directional trajectory of a trend. It brought clarification as a supervised tool. That worked then; however, for traders hunting geometric patterns, confirming a trend is not very helpful if the market's fundamental structure is actively collapsing. This in geometric markets stems from macroeconomic shocks as mentioned above, and therefore a structural shield would be better suited here, so we replace the camera filter with a compression artist.

To see how this works, imagine an artist who is meant to look at the recent geometric chart, compress it into a tiny restricted memory format, and then try to redraw it. If the current market's fractal points are following a preset rhythm, this artist would be able to easily capture this essence based on the past repetition principle. However, in situations where this rhythm breaks and becomes chaotic or unreadable, the artist's attempt to compress and redraw this resulting chaos is more likely to result in an unrecognizable mess. In this article, we want to use this 'messiness' (the math reconstruction error) as the final gatekeeper. It will be an anomaly detector, proactively reducing our risk as timely as possible, whenever geometric markets break structure. The goal here is to do this before there is a need to even use a stop-loss or alternative preset exit mechanisms.



Math Definition
Now we formalize the two checks to translate our above ideas into a workable MQL5 custom class. For a trader that hunts geometric patterns, our math objective can actually be two-fold. First we need to map exact discrete sequences in real-time, and secondly we need to measure the continuous geometric integrity of the market. This approach then is clearly different from say the reactive loss-counting approach with the built-in class.

The Suffix Automaton

In computational graph theory, a Suffix Automaton (SA) is formally defined as the least deterministic automaton that is finite while still recognizing all the possible suffixes of a given string. If we let S represent our historical 'Price DNA', a string of length n constructed solely from the categorical alphabet sigma = {U,D,F} or (Up, Down, Flat), its automaton is a Directed Acyclic Graph (DAG) where every node stands for a state, and every directed edge represents a deterministic transition triggered by a character from this preset alphabet.

A state in the SA will correspond to an equivalent class of substrings that share matching ending positions in the historical dataset (aka the 'endpos' equivalence class). In addition, every state has a 'suffix-link' - a structural pointer to another state that represents the longest suffix of the substrings that should be in different 'endpos' class. Why would this complicated discrete mathematics be important to a retail geometric trader? Because searching very deep historical arrays for an exact sequence using regular nested for loops can take up O(n^2) time complexity. In the MQL5 Expert Advisor environment that processes data multiple times within a given time interval O(n^2) scanning can easily freeze an execution thread or throttle quota VPS resources.

However, if we use Blumer's algorithm, the Suffix Automaton constructs its whole historical index online in a strictly linear time O(n) that has its limits defined by a maximum of 2n - 1 states and 3n - 4 transitions. If we then query the SA with a recent price sequence T, traversing the DAG assures us of an instant query time of O(|T|). We would be isolating the longest matching sequence from our price history immediately, and thus attaining a near-perfect categorical pattern match without exerting pressure on server resources.

The Autoencoder

While the Automaton manages discrete sequence pairing, the Autoencoder is charged with continuous anomaly detection in geometric patterns. Essentially an Autoencoder is an unsupervised feedforward neural network that is designed to learn an identity function - it tries to output the exact same data it receives, however it is forced to do so via a bottleneck. Supposing x is an input vector, that is normalized as a sequential array of recent raw prices. The neural network would be divided into two symmetrical parts. An encoder pass and a decoder pass. The encoder would compress raw market patterns into a hidden bottleneck layer, h, where its size is less than that of the input layer. This mandatory dimension reduction prevents the network from merely memorizing the input. It makes the network learn the principal geometric structures of the market data. The formal definition of an encoder is as follows:

f1

Where:

W(e) is the encoder (k x m) matrix
b(e) is the bias vector
sigma(z) is the non linear sigmoid activation function.
Following this, the decoder pass would try to rebuild the original market patterns from the highly compressed h layer. It's defined as follows:

f2

Where:

x is the reconstructed output vector
W(d) is the decoder (m × k) matrix
b(d) is the decoder bias
In order to proactively spot structural breaks in the market environment, we work out the 'reconstruction-loss' by using the Mean Squared Error between the real market input and the network's redraw by using the following formula:

f3

Finally, we need to translate this loss into something we can use in the format of a multiplier that is continuous for our geometric position sizing. We also set our 'Confidence Coefficient' C in [0, 1] as:
f4

The equation above serves as the geometric trader's safety net. It almost removes the need for reactive consecutive loss counting. Once the market is sound and in rhythm (no breaks), MSE would approach 1.0, which would allow the Automaton's lot size to be preserved, since no risk mitigating measures with lot size reduction would be necessary. Conversely, if some macroeconomic news or events shatter market's structure, the neural network would be unable to reconstruct this input vector of chaos. This would be marked with spikes in MSE, quickly inflating the denominator, meaning C would crash towards zero. Mathematically, it would be a choke off point for capital exposure as soon as this structural integrity is registered preventing the trade account from booking another loss.

MQL5 Implementantion
With our theory and math-engines laid out above, we can progress to the MQL5 coding. In order to have a seamless integration with MQL5 Wizard framework, as always, our custom class inherits from an appropriate standard library base class. In this case since we are building a money management class 'CMoneySuffixAE' and therefore we will inherit from 'CExpertMoney' base class. With that said, this is where most similarity our class will have to other built-in money management classes ends. Looking at the 'Optimize' function for instance we can spot how our new class evolves from the 'reactive-rearview' approach in favor of a strictly 'proactive-geometric-posture'.

Mini-map: What Optimize() does step by step

Before diving into the code, here is a quick overview of what the `Optimize()` function executes:

Look Outward: Abandons historical loss tracking and instead pulls the latest raw price data.
Build the Genetic Index: Dynamically constructs the Suffix Automaton from historical data.
Sequence the DNA: Extracts the recent discrete patterns (for the Automaton) and continuous raw prices (for the Autoencoder).
Query the Automaton: Evaluates the familiarity of the recent sequence and calculates a lot-size scaling factor based on the selected algorithm mode.
Pass the Gatekeeper: Calculates the reconstruction error using the Autoencoder to definitively slash risk if the market structure is unreadable.
Evolving from the Rearview Mirror

Let us briefly look at the core logic behind the built-in 'CMoneySizeOptimized' class. Within its 'Optimize()' we run the following routine:

if(m_decrease_factor > 0)
  {
   //--- select history for access
   HistorySelect(0, TimeCurrent());

   int orders = HistoryDealsTotal();
   int losses = 0;
   CDealInfo deal;

   for(int i = orders - 1; i >= 0; i--)
     {
      deal.Ticket(HistoryDealGetTicket(i));
      if(deal.Ticket() == 0)
        {
         Print("CMoneySizeOptimized::Optimize: HistoryDealGetTicket failed, no trade history");
         break;
        }

      //--- check symbol
      if(deal.Symbol() != m_symbol.Name())
         continue;

      //--- check profit
      double profit = deal.Profit();
      if(profit > 0.0)
         break;
      if(profit < 0.0)
         losses++;
     }

   if(losses > 1)
      lot = NormalizeDouble(lot - lot * losses / m_decrease_factor, 2);
  }
What we have here is the anatomy of a lagging indicator. We are explicitly requesting the account's historical ledger ('HistorySelect()'), iterating across closed order tickets, counting consecutive negative yields, in order to decide whether to reduce the input trade volume. This can demand that the geometric trader absorbs losses prior to this function being useful. Now, if we contrast this with the function we have below for our new class labelled 'CMoneySuffixAE::Optimize()' we can notice some changes:

class CMoneySuffixAE : public CExpertMoney
  {
protected:
   CSuffixAutomaton   m_sa;
   CAutoEncoder       m_ae;

   int                m_history_length;
   int                m_dna_window;
   int                m_algo_mode;
   bool               m_use_ae;

public:
                     CMoneySuffixAE(void);
                    ~CMoneySuffixAE(void);

   virtual double     CheckOpenLong(double price,double sl);
   virtual double     CheckOpenShort(double price,double sl);

protected:
   double             Optimize(double lots);
  };
The class uses two engines. The first engine is the Suffix Automaton. It works on a discrete representation of price movement. A higher close becomes U , a lower close becomes D , and an unchanged close becomes F . In the implementation these are encoded as integers.

int char_code = (diff > 0) ? 0 : ((diff < 0) ? 1 : 2); // 0=U, 1=D, 2=F
The second engine is the Autoencoder. It works on normalized raw prices and acts as a structural filter. If the recent price shape is still readable, the Autoencoder should reconstruct it with low error. If the market becomes chaotic, the reconstruction error rises and the lot size is reduced.

The rest of the article follows the implementation of Optimize() step by step. The explanatory text is therefore attached to the code path itself rather than presented as a large theoretical introduction.


Step 1: Pull Market Structure Instead of Trade History
The built-in class begins with HistorySelect() . Our class begins with CopyClose() .

double CMoneySuffixAE::Optimize(double lots)
  {
   double lot = lots;

   if(m_history_length > 0)
     {
      double closes[];
      ArraySetAsSeries(closes, true);

      if(CopyClose(m_symbol.Name(),
                   PERIOD_CURRENT,
                   0,
                   m_history_length + m_dna_window,
                   closes) == m_history_length + m_dna_window)
        {
         // further processing continues here
        }
     }

   return lot;
  }
This change is the main philosophical pivot of the class. The lot size is no longer adjusted according to the account’s last few closed deals. Instead, the class reads the current symbol’s price structure directly. This makes the money-management model outward-looking rather than inward-looking.

For momentum systems, a previous implementation used a Fenwick Tree with On-Balance Volume and a 1D CNN. That made sense because the central question was about cumulative volume pressure. Here the environment is different. A geometric trader does not primarily ask how much volume is pushing the market. The trader asks whether the current sequence of price action is still structurally familiar.


Step 2: Build the Historical Price DNA
Once the closing prices are available, the class resets the automaton and begins converting historical price movement into a discrete alphabet.

m_sa.Reset();

for(int i = m_dna_window;
    i < m_history_length + m_dna_window - 1;
    i++)
  {
   double diff = closes[i] - closes[i + 1];

   int char_code = (diff > 0) ? 0 :
                   ((diff < 0) ? 1 : 2); // 0=U, 1=D, 2=F

   m_sa.Extend(char_code);
  }
This is where the Suffix Automaton replaces the Fenwick Tree. The Fenwick Tree was useful when the task was to keep efficient cumulative totals. It behaved like an accountant: it could quickly tell us how much volume pressure had accumulated.

The Suffix Automaton is different. It behaves more like a genetic index. It does not care about totals. It cares about sequences. A simplified Price DNA string might look like this:

U D U U F D D U F U
For a geometric trader, this is closer to the actual trading question. The system wants to know whether the current sequence of price movement has appeared before, and how deeply it matches the historical structure. A naive historical search could become expensive if implemented through nested loops over large arrays. The Suffix Automaton avoids this by building a compact structure that can be queried efficiently.


Step 3: Extract the Current Setup
The automaton is built from historical data. Now the class needs the current pattern immediately preceding the trade signal. That is done by filling two arrays at the same time.

int recent_dna[];
double recent_raw[];

ArrayResize(recent_dna, m_dna_window);
ArrayResize(recent_raw, m_dna_window);

double max_price = closes[0];
double min_price = closes[0];

for(int i = 0; i < m_dna_window; i++)
  {
   double diff = closes[m_dna_window - 1 - i] -
                 closes[m_dna_window - i];

   recent_dna[i] = (diff > 0) ? 0 :
                   ((diff < 0) ? 1 : 2);

   recent_raw[i] = closes[m_dna_window - 1 - i];

   if(recent_raw[i] > max_price)
      max_price = recent_raw[i];

   if(recent_raw[i] < min_price)
      min_price = recent_raw[i];
  }
recent_dna is the discrete version of the setup. It is passed to the Suffix Automaton and answers the question: Does the current directional sequence have a historical twin?

recent_raw preserves the actual price shape. It is later passed to the Autoencoder and answers a different question: Even if the sequence looks familiar, is the current market structure still clean enough to trust?

This separation is important. A discrete sequence can look familiar while the underlying price behavior is already unstable. For example, a pattern may still produce the same U-D-U-D signature during a macroeconomic shock, but the shape, range, and volatility of that movement may no longer be normal. The Suffix Automaton sees sequence familiarity. The Autoencoder sees structural integrity.


Step 4: Query the Suffix Automaton
After preparing the recent DNA sequence, the class asks the automaton for the longest historical match.

int longest_match = m_sa.GetLongestMatch(recent_dna);

double dna_score = (double)longest_match /
                   (double)m_dna_window;

double scale_factor = 1.0;

switch(m_algo_mode)
  {
   case 1:
      scale_factor = AlgorithmMode1_Linear(dna_score);
      break;

   case 2:
      scale_factor = AlgorithmMode2_Conservative(dna_score);
      break;

   case 3:
      scale_factor = AlgorithmMode3_Aggressive(dna_score);
      break;

   case 4:
      scale_factor = AlgorithmMode4_MeanReversion(dna_score);
      break;
  }

lot = NormalizeDouble(lot * scale_factor, 2);
The result is normalized into dna_score , a value between 0.0 and 1.0 . A value near zero means the current sequence is mostly unfamiliar. A value near one means the recent setup has a strong historical match. This score becomes the first position-sizing signal.

Different geometric traders may interpret pattern familiarity differently. A breakout trader may want to increase size when a known pattern is strongly confirmed. A contrarian trader may prefer to do the opposite when the pattern looks too obvious or exhausted. For that reason, the class exposes several algorithm modes.


Step 5: Translate Pattern Familiarity into Risk
The algorithm modes are only money-management curves. They do not generate entries by themselves. The entry signal still comes from the Expert Advisor assembled by the MQL5 Wizard. The modes simply convert dna_score into a multiplier.

double CMoneySuffixAE::AlgorithmMode1_Linear(double score)
  {
   return 0.8 + 1.2 * score;
  }

double CMoneySuffixAE::AlgorithmMode2_Conservative(double score)
  {
   return 0.5 + MathSqrt(score) * 0.75;
  }

double CMoneySuffixAE::AlgorithmMode3_Aggressive(double score)
  {
   return 0.5 + 2.5 * score * score;
  }

double CMoneySuffixAE::AlgorithmMode4_MeanReversion(double score)
  {
   if(score <= 0.0)
      return 1.0;

   return MathMin(3.0, 0.5 / score);
  }
The first mode is linear. It is the most direct interpretation of the DNA score: unfamiliar structures reduce position size, while familiar structures increase it.

The second mode is conservative. It is more suitable for traders who want pattern confirmation but do not want aggressive lot expansion. The square-root curve responds early to moderate confidence while keeping total scaling restrained.

The third mode is aggressive. It heavily rewards high-confidence matches and is designed for situations where the trader only wants to scale meaningfully when the pattern match is strong.

The fourth mode uses inverse logic. It is intended for mean-reversion or contrarian interpretations. Instead of rewarding perfect familiarity, it assumes that extremely obvious structure may be exhausted.


Step 6: Add the Autoencoder Gatekeeper
The Suffix Automaton gives us a pattern-familiarity score, but it has one limitation: it sees discrete direction, not continuous distortion. That is where the Autoencoder is added.

if(m_use_ae)
  {
   double norm_range = (max_price - min_price == 0)
                       ? 1.0
                       : (max_price - min_price);

   for(int i = 0; i < m_dna_window; i++)
      recent_raw[i] = (recent_raw[i] - min_price) / norm_range;

   double ae_coefficient = m_ae.Calculate(recent_raw);

   lot = NormalizeDouble(lot * ae_coefficient, 2);
  }
The class first normalizes the recent raw price window into the [0, 1] range. This gives the neural network a stable input scale. Then the normalized price structure is passed into the Autoencoder.

The Autoencoder compresses the input into a smaller hidden representation and tries to reconstruct the original input from that compressed form. If the recent market structure resembles the type of structure the network can represent, reconstruction error should be low. If the market has become chaotic, the reconstruction error should rise.

h = σ(W e x + b e )

x′ = σ(W d h + b d )

MSE = (1 / n) × Σ(x i − x′ i )2

C = 1 / (1 + MSE)

When reconstruction error is small, C remains close to 1.0 , so the Suffix Automaton’s lot decision is mostly preserved. When reconstruction error rises, C moves toward zero, and the position size is reduced.

This gives the class a second layer of defense. The automaton asks whether the sequence is familiar. The Autoencoder asks whether the current structure is still clean enough to trade with normal risk.


Step 7: Validate Broker Limits
After both engines have adjusted the lot size, the class still needs to respect the broker’s volume constraints.

double stepvol = m_symbol.LotsStep();
lot = stepvol * NormalizeDouble(lot / stepvol, 0);

double minvol = m_symbol.LotsMin();
if(lot < minvol)
   lot = minvol;

double maxvol = m_symbol.LotsMax();
if(lot > maxvol)
   lot = maxvol;

return lot;
This final block is not part of the model’s intelligence, but it is essential for execution. No matter how the Suffix Automaton or Autoencoder scales the position, the resulting lot size must be aligned with the symbol’s volume step and remain within the broker’s minimum and maximum limits.

At this point the full Optimize() workflow can be summarized as follows:

1. Read recent and historical Close prices.
2. Convert historical movement into Price DNA.
3. Build the Suffix Automaton.
4. Convert the current setup into recent DNA.
5. Query the automaton for the longest historical match.
6. Convert the match score into a lot multiplier.
7. Optionally pass the raw price structure through the Autoencoder.
8. Reduce lot size if reconstruction error is high.
9. Normalize the result to broker constraints.

The Full Optimize Logic
The core routine can therefore be read as a single pipeline. The order matters: first the Suffix Automaton proposes a multiplier based on historical sequence familiarity, and only then the Autoencoder is allowed to suppress that multiplier if the current structure is distorted.

double CMoneySuffixAE::Optimize(double lots)
  {
   double lot = lots;

   if(m_history_length > 0)
     {
      double closes[];
      ArraySetAsSeries(closes, true);

      //--- Pull enough bars for both the historical index and the current DNA window
      if(CopyClose(m_symbol.Name(),
                   PERIOD_CURRENT,
                   0,
                   m_history_length + m_dna_window,
                   closes) == m_history_length + m_dna_window)
        {
         m_sa.Reset();

         //--- 1. Build the Suffix Automaton from historical Price DNA
         for(int i = m_dna_window;
             i < m_history_length + m_dna_window - 1;
             i++)
           {
            double diff = closes[i] - closes[i + 1];
            int char_code = (diff > 0) ? 0 :
                            ((diff < 0) ? 1 : 2); // 0=U, 1=D, 2=F

            m_sa.Extend(char_code);
           }

         //--- 2. Extract the current setup in both discrete and raw form
         int recent_dna[];
         double recent_raw[];

         ArrayResize(recent_dna, m_dna_window);
         ArrayResize(recent_raw, m_dna_window);

         double max_price = closes[0];
         double min_price = closes[0];

         for(int i = 0; i < m_dna_window; i++)
           {
            double diff = closes[m_dna_window - 1 - i] -
                          closes[m_dna_window - i];

            recent_dna[i] = (diff > 0) ? 0 :
                            ((diff < 0) ? 1 : 2);

            recent_raw[i] = closes[m_dna_window - 1 - i];

            if(recent_raw[i] > max_price)
               max_price = recent_raw[i];

            if(recent_raw[i] < min_price)
               min_price = recent_raw[i];
           }

         //--- 3. Query the automaton and convert the match into a score
         int longest_match = m_sa.GetLongestMatch(recent_dna);
         double dna_score = (double)longest_match /
                            (double)m_dna_window;

         //--- 4. Convert the DNA score into a lot multiplier
         double scale_factor = 1.0;

         switch(m_algo_mode)
           {
            case 1:
               scale_factor = AlgorithmMode1_Linear(dna_score);
               break;

            case 2:
               scale_factor = AlgorithmMode2_Conservative(dna_score);
               break;

            case 3:
               scale_factor = AlgorithmMode3_Aggressive(dna_score);
               break;

            case 4:
               scale_factor = AlgorithmMode4_MeanReversion(dna_score);
               break;
           }

         lot = NormalizeDouble(lot * scale_factor, 2);

         //--- 5. Use the Autoencoder as a structural gatekeeper
         if(m_use_ae)
           {
            double norm_range = (max_price - min_price == 0)
                                ? 1.0
                                : (max_price - min_price);

            for(int i = 0; i < m_dna_window; i++)
               recent_raw[i] = (recent_raw[i] - min_price) / norm_range;

            double ae_coefficient = m_ae.Calculate(recent_raw);

            lot = NormalizeDouble(lot * ae_coefficient, 2);
           }
        }
     }

   //--- 6. Validate broker limits
   double stepvol = m_symbol.LotsStep();
   lot = stepvol * NormalizeDouble(lot / stepvol, 0);

   double minvol = m_symbol.LotsMin();
   if(lot < minvol)
      lot = minvol;

   double maxvol = m_symbol.LotsMax();
   if(lot > maxvol)
      lot = maxvol;

   return lot;
  }
This order is important because a familiar sequence is not always a safe sequence. A pattern can still look familiar in discrete form while the continuous market structure is already breaking.


Post-Optimization Analysis
To see whether each part of the model contributes something useful, the class was tested in two configurations on GBPUSD using the H4 timeframe. The optimization period was 2025, and the forward walk covered 2026.01.01 to 2026.05.01.

The first run used only the Suffix Automaton. The automaton was kept in Mode 1, while the Autoencoder filter was disabled.

m_algo_mode = 1
m_use_ae    = false
This test checks whether sequence familiarity alone can provide a usable money-management signal. The forward walk produced the following report.

r1

The result shows that the automaton can produce a profitable baseline. That matters because the lot adjustment was not based on recent account losses. It was based on whether the current Price DNA had historical similarity. However, the equity curve remained volatile, with drawdown approaching 11%.

That weakness is consistent with the limitation discussed earlier. A Suffix Automaton can recognize sequence familiarity, but it does not necessarily recognize structural chaos. It can tell us that a recent sequence resembles history, but it cannot fully judge whether the current market environment has been distorted by a sudden shock.

The second run added the Autoencoder.

m_algo_mode = 1
m_use_ae    = true
Now the position size was first adjusted by the Suffix Automaton and then filtered by the Autoencoder’s reconstruction coefficient.

r2

This forward result was more stable, with the equity drawdown almost halved compared with the automaton-only run. This does not prove that the model is universally robust. The test was performed on one symbol and one limited time window. Additional tests across symbols, timeframes, spreads, and market regimes are required before any adoption should be considered.

Still, the comparison is useful. It suggests that the two engines are not redundant. The Suffix Automaton contributes pattern familiarity. The Autoencoder contributes structural defense.


Practical Takeaways
The main lesson from this implementation is that money management does not have to be based only on the account’s trade history. The built-in CMoneySizeOptimized class reduces volume after a sequence of losses. That approach is simple and practical, but it is reactive. The market must first inflict damage before the class responds.

The CMoneySuffixAE class tries to respond earlier. It uses current market structure rather than closed trade history. For momentum-oriented systems, volume-based methods such as a Fenwick Tree applied to OBV can make sense. For geometric or pattern-based systems, sequence familiarity may be more relevant. That is where the Suffix Automaton becomes useful.

However, sequence familiarity alone is not enough. A market can still produce recognizable directional fragments while its broader structure is breaking. The Autoencoder is added to catch that kind of distortion by measuring reconstruction error.

The resulting model is therefore conditional. If the current sequence is familiar and the structure is clean, the class can preserve or increase exposure. If the sequence is unfamiliar or the structure becomes unreadable, the class can reduce exposure before the loss counter would normally react.


Conclusion
This article built a custom MQL5 Wizard money-management class named CMoneySuffixAE . Instead of reducing position size after a series of losing trades, the class reads the current market structure directly. It converts historical price action into discrete Price DNA, uses a Suffix Automaton to measure how familiar the current setup is, and then applies an Autoencoder as a structural gatekeeper.

The result is a proactive money-management model designed for geometric traders — traders whose systems depend on recognizable price structures rather than cumulative volume pressure. The tests suggest that the Suffix Automaton can provide a useful sequence-based sizing signal, while the Autoencoder can help reduce drawdown by suppressing exposure during structurally unstable periods.

The broader idea is simple: for pattern-based systems, risk should not wait for the account history to confirm that something has gone wrong. Sometimes the market structure gives the warning first.

name	description
wz_93.mqh	Wizard Assembled Expert Advisor (links shared for new readers)
MoneySizeOptimized.mqh	Built-In sampled Custom Money Managment Class
MoneySuffixAE.mqh	Article Coded Custom Money Management class
r1.set	Input settings for Expert Advisor in run 1
r2.set	Input settings for Expert Advisor in run 2

---

## Ideas From This Article We Have NOT Done Yet

### 1. Algo Mode Comparison (Modes 2, 3, 4)

We implemented all four modes but only backtested Mode 1 (linear). The article describes three others with meaningfully different risk profiles:

```
Mode 2 — Conservative: scale_factor = 0.5 + sqrt(score) * 0.75   range [0.5, 1.25]
Mode 3 — Aggressive:   scale_factor = 0.5 + 2.5 * score^2        range [0.5, 3.0]
Mode 4 — Mean-Rev:     scale_factor = min(3.0, 0.5 / score)       rewards unfamiliar patterns
```

```bash
conda run -n envmt5 python scripts/backtest_suffix_ae.py --algo-mode 2
conda run -n envmt5 python scripts/backtest_suffix_ae.py --algo-mode 3
conda run -n envmt5 python scripts/backtest_suffix_ae.py --algo-mode 4
conda run -n envmt5 python scripts/backtest_suffix_ae.py --no-ae
```

### 2. SA-Only vs AE-Only Isolation Test

The article claims the two engines are not redundant. We have not verified this on our data.

- SA-only (`--no-ae`): pattern familiarity drives sizing, no structural gate
- AE-only: modify `SuffixAESizer` to bypass SA and pass `scale_factor=1.0` through AE only
- Expected from article: AE gate roughly halves drawdown vs SA-only on GBPUSD H4
- Our result may differ on EURUSD M15 — worth confirming before relying on AE in live trading

### 3. Multi-Symbol / Forward Walk Validation

The article tested on GBPUSD H4 with 2025 in-sample and 2026.01-2026.05 forward.
We only ran EURUSD M15 across the full 2-year window (no true forward walk split).

```bash
conda run -n envmt5 python scripts/backtest_suffix_ae.py --symbol USDJPY
conda run -n envmt5 python scripts/backtest_suffix_ae.py --symbol GBPUSD
```

For a proper forward walk: train pipeline on 2024-2025 only, test SA+AE on 2026 only.
This is the validation the article used to claim "drawdown almost halved."

---

## Related Article — MQL5 Wizard (Previous): Fenwick Tree + OBV + 1D CNN Money Management

Referenced in wz_93 as "our last outing" — the momentum-trader counterpart to CMoneySuffixAE.

Core idea: Instead of Price DNA sequences, use On-Balance Volume (OBV) as the input signal.

- **Fenwick Tree** (Binary Indexed Tree): efficient cumulative OBV prefix sums, O(log n) per update
- **1D CNN**: sliding window over OBV data to smooth noise and identify directional trajectory
- **Position sizing**: scale lots based on cumulative volume pressure (momentum confirmation)

When it beats SA+AE: trending markets with strong volume confirmation. SA+AE is blind to volume.
When SA+AE beats it: geometric/pattern markets, structural breaks, news-driven volatility.

```python
class FenwickTree:
    def update(self, i, delta): ...  # O(log n)
    def query(self, i): ...          # prefix sum [0..i], O(log n)

class OBVMoneySizer:
    def compute(self, closes, volumes) -> float:
        # 1. Build Fenwick tree over OBV increments
        # 2. Extract OBV window, normalize
        # 3. Run 1D conv smoothing (numpy correlate)
        # 4. Scale by trend strength of smoothed OBV
```

TO DO: Implement `OBVMoneySizer` in `src/core/obv_sizer.py`, add to `BacktestConfig`, compare against `SuffixAESizer`.

---

## Idea — Triple-Barrier Meta-Labeling (Phase 26 in roadmap)

Source: Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018), Chapter 3.

Problem it solves: our current labels are fixed SL/TP pips. Triple-barrier labeling adds a time
limit (vertical barrier) and captures the path, not just the outcome.

```
Upper barrier: entry + tp_pips  -> label = +1 (buy wins)
Lower barrier: entry - sl_pips  -> label = -1 (sell wins)
Time barrier:  entry + n_bars   -> label =  0 (hold/expire)
```

Meta-labeling layer on top: train a second model that predicts P(primary_model_is_right).
This filters out low-quality signals regardless of direction. Our win rate is ~37.8% —
meta-labeling typically moves this to 45-55% without changing the primary signal at all.

TO DO (Phase 26):

1. Add `triple_barrier_labels()` to `src/features/feature_pipeline.py`
2. Train a second XGBoost on primary model's `P_buy/P_sell` + features -> `P(correct)`
3. Gate trades: only execute when `meta_confidence > 0.55`
4. Run walk-forward: compare win rate and Sharpe vs baseline

---

## Idea — Hidden Markov Model Regime Detection (Phase 25 in roadmap)

Source: well-established in quant finance; used by Two Sigma, AQR for regime-switching models.

Problem it solves: our enc8 XGBoost was trained on all 60k bars — bull, bear, ranging, trending.
A model trained on everything is optimal for nothing. Regime detection splits the market into
states and uses a dedicated model per state.

```
Features: rolling returns, volatility, ATR, ADX
States:   trending (directional + low vol), ranging (choppy + high vol), crisis
Output:   P(regime_k | observations_1..t) — posterior per bar
```

How it routes into our system:

- `regime=trending` -> XGBoost enc8 (current champion)
- `regime=ranging`  -> hold, or `SuffixAESizer mode=4` (mean-reversion sizing)
- `regime=crisis`   -> `SuffixAESizer` AE coefficient naturally slashes lots

TO DO (Phase 25): `pip install hmmlearn` then `src/features/regime_detector.py` —
`HMMRegimeDetector` class. Add `regime_id` as a feature column so XGBoost learns
regime-conditional patterns.

---

## Idea — Multi-Timeframe Feature Stacking (Phase 24 in roadmap)

Referenced in IMPROVEMENT.MD under "External Features — Multi-Timeframe."

Problem it solves: M15 features are blind to the H1/H4 trend context. A buy signal on M15 that
goes against the H4 trend has lower probability of success.

```
H1: SMA_20, EMA_9, RSI_14, ATR_14, ADX_14 (aligned to M15 index via ffill)
H4: SMA_20, trend_slope (linear regression over last 10 H4 bars), ADX_14
D1: session_high, session_low, daily_range_pct
```

Data files already available: `data/EURUSD_H1.csv`, `data/EURUSD_H4.csv`, `data/EURUSD_D1.csv`

Our enc8 feature saturation finding confirms the ceiling on M15-only features. Multi-timeframe
is the lowest-effort next step since the data is already downloaded.

TO DO (Phase 24): `src/features/mtf_features.py` — `MTFFeatureBuilder` class. Adds ~15 columns,
no architecture changes. Run `compare_h1_vs_m15.py` (already in `scripts/`) to validate.

---

## Idea — Contrastive Self-Supervised Encoder Pre-training (Technique 5 in IMPROVEMENT.MD)

Source: SimCLR (Chen et al., 2020), applied to finance by JPMorgan and Meta's quant teams.

Problem it solves: our enc8 only knows 3 classes (buy/hold/sell). Contrastive pre-training teaches
the encoder a richer representation WITHOUT labels — similar market regimes produce similar latent
vectors, different regimes produce different vectors.

How it works:

1. Take a 50-bar OHLCV window (anchor)
2. Create two augmented views: add noise, drop 20% of bars, scale by +/-5%
3. Train encoder so anchor and its views are close in latent space (positive pairs)
4. Train encoder so anchor and unrelated windows are far apart (negative pairs)
5. Loss: NT-Xent (normalized temperature-scaled cross-entropy)

After pre-training: fine-tune the encoder head with our existing supervised labels. Expected
benefit: latent vectors distinguish "quiet trend" from "noisy trend" from "compression" even
when all three map to the same "buy" label.

TO DO: `src/models/contrastive_encoder.py` — `ContrastiveEncoder` with NT-Xent loss.
Pre-train on all 60k bars (no labels needed), then fine-tune to replace enc8.

---

## Idea — Online Walk-Forward Adaptation (Technique 3 in IMPROVEMENT.MD)

Source: "Machine Learning for Asset Managers" (Lopez de Prado, 2020).

Problem it solves: our champion model drifts between weekly retrains. Online adaptation updates
weights continuously using recent data without a full retrain.

Three options ranked by risk:

- **Option A — Sliding calibration**: re-fit only `CalibratedClassifierCV` on the last 500 bars
  each day. XGBoost base stays frozen. Cost: ~2 seconds.
- **Option B — Warm-start boosting**: add 1 new XGBoost tree per day fitted on recent errors
  via `model.get_booster()` incremental fitting.
- **Option C — Snapshot blending**: keep 3 model snapshots (1-week, 1-month, 3-month) and blend
  with weights proportional to recent Sharpe. No retraining needed. Lowest risk entry point.

TO DO: `src/models/adaptive_ensemble.py` — `AdaptiveEnsemble` with rolling Sharpe weighting.
Run `compare_adaptive_wf.py` (already in `scripts/`) to validate.

---

## Summary Table — What To Do Next

| name | description | status |
|---|---|---|
| Mode 2/3/4 backtest | `backtest_suffix_ae.py --algo-mode 2/3/4` | not started — easy (30 min) |
| SA-only isolation | `--no-ae` to isolate SA contribution | not started — easy (15 min) |
| Multi-symbol SA+AE | Test SA+AE on USDJPY and GBPUSD | not started — easy (1 hr) |
| OBV Fenwick sizer | Python port of the Fenwick Tree + OBV article (wz_92) | not started — medium (1 day) |
| Multi-timeframe features | H1/H4/D1 features aligned to M15 (Phase 24) | not started — medium (1 day) |
| Triple-barrier labels | 3-barrier labels + meta-model layer (Phase 26) | not started — medium (2 days) |
| HMM regime detection | 2-state HMM to route signals per regime (Phase 25) | not started — medium (1 day) |
| Adaptive ensemble | Rolling Sharpe-weighted model snapshot blending | not started — medium (1 day) |
| Contrastive encoder | SimCLR-style pre-training to replace enc8 (Technique 5) | not started — hard (3 days) |


---

## What Can We Use From READ 7 (Fractal Patterns ML)?

**READ 7** is the most directly usable article added. It implements exactly the kind of labeling + meta-model approach mentioned in our "Triple-Barrier Meta-Labeling" section, but using fractal symmetry instead of fixed barriers.

### Core idea

A symmetric correlation scan identifies self-similar price structures (fractals) in the data. Bars inside a high-correlation fractal pattern are labeled based on future price direction. Two CatBoost models are then trained:

- **Main model** — predicts direction (buy=0 / sell=1) for bars inside fractal patterns only
- **Meta-model** — predicts whether to trade at all (1 = inside a valid fractal, 0 = skip)

This is structurally identical to meta-labeling but driven by fractal structure rather than a time barrier.

### Key function to port

```python
# @njit (Numba) — finds best symmetric correlation for each bar
def calculate_symmetric_correlation_dynamic(data, min_window, max_window):
    # For each starting point i, checks windows of size min_w..max_w (even only)
    # Splits window in half, mirrors+inverts right half, computes Pearson correlation
    # Returns: correlations[], best_window_sizes[]
    ...

# Labels trades based on high-correlation fractal patterns
def get_fractal_pattern_labels(dataset, min_window=6, max_window=60,
                               corr_threshold=0.7, min_horizon=5,
                               max_horizon=5, markup=0.0001):
    # Returns dataset with 'labels' column: 0=buy, 1=sell, 2=no-trade
    ...
```

### Connection to our existing work

| Concept | READ 7 fractal | Our SA+AE |
|---|---|---|
| Pattern detection | Symmetric correlation (Pearson r) | Suffix Automaton (sequence match) |
| Structural guard | Correlation threshold (r > 0.7/0.9) | Autoencoder reconstruction error |
| Output | Training labels (direction + trade/no-trade) | Lot-size multiplier at runtime |
| When applied | Offline (label generation) | Online (live sizing each trade) |

They are complementary: fractal labeling improves **what the model learns**, SA+AE improves **how much it bets** on each signal.

### Article result (EURUSD H1, 2010–2024 train, 2024+ forward)

- Main model validation accuracy: 97%
- Meta-model F1: 0.85
- Forward test: profitable equity curve (see Fig. 7–8 in article)
- Best window: 100 bars, correlation threshold: 0.9

### What to build

```
src/features/fractal_labeler.py   — FractalLabeler class
  - calculate_symmetric_correlation()  (use numba if available, else numpy fallback)
  - get_fractal_labels(df, min_w, max_w, corr_threshold, horizon, markup)
  - Returns df with labels column: 0 / 1 / 2

src/models/meta_classifier.py     — MetaClassifier (already partially exists as concept)
  - Wraps primary model + meta-model
  - Predicts: direction AND whether to trade
  - Returns P_buy, P_hold, P_sell where P_hold is elevated when meta says skip

scripts/train_fractal_model.py    — train fractal-labeled CatBoost + meta-model
```

### Comparison to what we have

Our current labels: fixed SL/TP forward-looking (buy if price rises by tp_pips in sl_pips time).
Fractal labels: directional label only for bars inside high-symmetry fractal structures.

The key difference: fractal labels are **structure-selective** — only ~20-40% of bars get a 0/1 label (the rest are 2=no-trade). This naturally produces fewer but higher-quality signals, which is exactly what meta-labeling aims for.

TO DO:

1. `pip install numba` (or use numpy fallback)
2. Implement `FractalLabeler` in `src/features/fractal_labeler.py`
3. Replace `feature_pipeline.py` label generation with fractal labels
4. Train two CatBoost models: direction model + trade/no-trade meta-model
5. Wire meta-model into `PipelineBot.on_tick()` as an additional gate before opening trades
6. Backtest: compare fractal-labeled model vs current fixed-SL/TP labels

---

## What Can We Use From READ 10 (Billiards Optimization Algorithm — BOA)?

**READ 10** (BOA) and **READ 11** (SRA) are population-based optimizers. The use case for us is **hyperparameter tuning** — replacing manual grid search with an automated optimizer.

### BOA in one equation (modified version — 62.19% score, rank 8)

```python
# Each "ball" moves toward a randomly chosen "pocket" (best solution so far)
X_new = X + random(0, 1) * (Pocket - X) * random_choice(1, 2)
```

That is the entire algorithm. The modification that doubles performance (31% → 62%): factor `I` out of the bracket so it scales the **difference**, not the absolute position.

### Where to use BOA in our project

| What to optimize | Parameters | Dimensions |
|---|---|---|
| XGBoost hyperparams | n_estimators, max_depth, learning_rate, subsample, colsample_bytree | 5 |
| SuffixAESizer params | history_length, dna_window, algo_mode | 3 |
| Backtest SL/TP | sl_pips, tp_pips, threshold | 3 |
| Fractal labeler | min_window, max_window, corr_threshold, horizon | 4 |

All of these are small/medium dimension problems (3–5 dims) — exactly where BOA performs best.

### Python implementation (the full algorithm — ~30 lines)

```python
class BilliardOptimizer:
    def __init__(self, n_agents=50, n_pockets=25, n_iters=100):
        self.n_agents  = n_agents
        self.n_pockets = n_pockets
        self.n_iters   = n_iters

    def optimize(self, objective_fn, bounds):
        # bounds: list of (min, max) per dimension
        import random, numpy as np
        dim = len(bounds)
        # Initialize agents randomly
        agents = np.array([[random.uniform(b[0], b[1]) for b in bounds]
                           for _ in range(self.n_agents)])
        best_agents = agents.copy()
        scores = np.array([objective_fn(a) for a in agents])
        best_scores = scores.copy()
        global_best = agents[scores.argmax()].copy()

        for _ in range(self.n_iters):
            # Top n_pockets become targets
            pocket_ids = np.argsort(best_scores)[-self.n_pockets:]
            for i, agent in enumerate(agents):
                p_id = random.choice(pocket_ids)
                pocket = best_agents[p_id]
                I = random.choice([1, 2])
                agent_new = agent + random.random() * (pocket - agent) * I
                # Clip to bounds
                agent_new = np.clip(agent_new,
                                    [b[0] for b in bounds],
                                    [b[1] for b in bounds])
                score = objective_fn(agent_new)
                if score > best_scores[i]:
                    best_agents[i] = agent_new
                    best_scores[i] = score
                if score > objective_fn(global_best):
                    global_best = agent_new.copy()
            agents = best_agents.copy()

        return global_best, best_scores.max()
```

### TO DO

```
src/core/billiard_optimizer.py  — BilliardOptimizer class (above)
scripts/optimize_pipeline.py    — tune XGBoost + SA+AE params using BOA
scripts/optimize_fractal.py     — tune fractal labeler params using BOA
```

---

## What to Skip (READ 11 SRA, READ 12 NOA)

**READ 11 — SRA (Successful Restaurateur Algorithm)**: rank 20, 54.48%. Less efficient than BOA (rank 8, 62.19%) for the same problem class. Skip — use BOA instead.

**READ 12 — NOA (Neuroboids Optimization Algorithm)**: last place (29.94%), worse than random walk on 1000-variable problems. The neural-network-per-agent approach is interesting conceptually but computationally impractical for any real use. Skip.

---

## Updated Summary Table (Including New READs)

| name | description | status |
|---|---|---|
| Mode 2/3/4 backtest | `backtest_suffix_ae.py --algo-mode 2/3/4` | not started — easy (30 min) |
| SA-only isolation | `--no-ae` to isolate SA contribution | not started — easy (15 min) |
| Multi-symbol SA+AE | Test SA+AE on USDJPY and GBPUSD | not started — easy (1 hr) |
| **Fractal labeler** | `FractalLabeler` in `src/features/fractal_labeler.py` (READ 7) | **not started — medium (2 days)** |
| **BOA optimizer** | `BilliardOptimizer` for hyperparameter tuning (READ 10) | **not started — easy (half day)** |
| OBV Fenwick sizer | Python port of Fenwick Tree + OBV (wz_92) | not started — medium (1 day) |
| Multi-timeframe features | H1/H4/D1 features aligned to M15 (Phase 24) | not started — medium (1 day) |
| Triple-barrier labels | 3-barrier labels + meta-model layer (Phase 26) | not started — medium (2 days) |
| HMM regime detection | 2-state HMM to route signals per regime (Phase 25) | not started — medium (1 day) |
| Adaptive ensemble | Rolling Sharpe-weighted model snapshot blending | not started — medium (1 day) |
| Contrastive encoder | SimCLR-style pre-training to replace enc8 (Technique 5) | not started — hard (3 days) |


READ 2

Introduction
In the previous article, we introduced the innovative CATCH framework for anomaly detection in multivariate time series. The frequency-domain analysis method proposed by the framework's authors enables not only the detection of point anomalies such as outliers and abrupt changes, but also the identification of more complex hidden patterns that often evade traditional approaches. The CATCH framework employs the Fourier transform to convert data from the time domain into its spectral representation, opening new opportunities for a detailed analysis of the characteristics of the sequence under examination.

One of the key advantages of CATCH is its use of frequency patching. Instead of analyzing the entire spectrum as a whole, the model divides it into separate fragments corresponding to specific frequency ranges. This approach makes it possible to examine both global trends and local features associated with high-frequency components. As a result, various types of anomalies can be identified and classified with high precision, whether they are short-term spikes or more complex deviations at the subsequence level. Such spectral partitioning enables a much more detailed study of the time-series structure.

An important component of the framework is the adaptive module responsible for analyzing relationships between different data channels. This component, which implements a masked attention mechanism, allows the system to focus on the most significant correlations while filtering out noise and irrelevant information. This mechanism improves the reconstruction quality of normal behavior and significantly enhances model robustness in a changing market environment. Unlike traditional methods that process channels independently, CATCH takes their complex interdependencies into account. This is particularly important when working with financial data, where movements in one market can influence other segments.

At the final stage of the CATCH framework, the original time series is reconstructed. After detailed frequency-domain analysis and the identification of potential anomalies, the system performs an inverse transformation, returning the data to its conventional time-domain representation. The difference between the original values and the reconstructed time series serves as a reliable indicator of deviations, enabling a timely response to changes in market dynamics.

The authors' visualization of the CATCH framework is shown below.

Author's visualization of the CATCH framework

In the practical section of the previous article, we began implementing our own interpretation of the proposed approaches using MQL5. Specifically, we developed a convolutional layer for processing complex-valued data. We also examined the implementation of the forward and backward passes of the complex-valued masked attention module within an OpenCL context. Today, we continue this work.


Complex-Valued Masked Attention Object
In the previous article, we examined the MaskAttentionComplex and MaskAttentionGradientsComplex kernels, which implement the forward and backward propagation algorithms of the masked attention mechanism in the complex domain. Today, we continue this work. We will organize the masked attention module on the main application side. To achieve this, we create a new object CNeuronComplexMVMHMaskAttention having the following structure:

class CNeuronComplexMVMHMaskAttention :  public CNeuronBaseOCL
  {
protected:
   uint                       iWindow;
   uint                       iWindowKey;
   uint                       iHeads;
   uint                       iUnits;
   uint                       iVariables;
   //---
   CNeuronComplexConvOCL      cQKV;
   CNeuronBaseOCL             cQ;
   CNeuronBaseOCL             cKV;
   CNeuronConvOCL             cMask;
   CNeuronBaseOCL             cMHAttentionOut;
   CNeuronComplexConvOCL      cPooling;
   CNeuronBaseOCL             cResidual;
   CNeuronComplexConvOCL      cFeedForward[2];
   //---
   virtual bool               AttentionOut(void);
   virtual bool               AttentionInsideGradients(void);
   virtual bool               SumAndNormilize(CBufferFloat *tensor1, CBufferFloat *tensor2, CBufferFloat *out,
                                              int dimension, bool normilize = true, int shift_in1 = 0,
                                              int shift_in2 = 0, int shift_out = 0, float multiplyer = 0.5f) override;
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;

public:
                     CNeuronComplexMVMHMaskAttention(void) {};
                    ~CNeuronComplexMVMHMaskAttention(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint window, uint window_key, uint heads, 
                                   uint units_count, uint variables, 
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void)   const   {  return defNeuronComplexMVMHMaskAttention;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
As can be seen, the structure of this new object is fairly standard for attention modules, several variations of which have already been implemented in our library. However, there is one important distinction associated with the use of complex-valued data. The object is designed to receive input data that has already been converted into complex form, allowing us to omit the data transformation stage. To generate the Query, Key, and Value representations, we use the previously developed convolutional layer for complex-valued processing. Let us examine the implementation step by step.

All internal objects are declared as class members, allowing us to keep the class constructor and destructor empty. The initialization of these declared and inherited objects is performed in the Init method. This method receives a set of constants that uniquely define the architecture of the object being created. Their structure is fairly typical for such objects.

bool CNeuronComplexMVMHMaskAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                           uint window, uint window_key, uint heads,
                                           uint units_count, uint variables,
                                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
//---
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, 2 * window * units_count * variables,
                                                                      optimization_type, batch))
      return false;
At the beginning of the method, we call the corresponding method of the parent class, which already contains the initialization logic for inherited objects and interfaces. We then store the model architecture parameters received from the external program in internal variables.

   iWindow = window;
   iWindowKey = MathMax(window_key, 1);
   iUnits = units_count;
   iHeads = MathMax(heads, 1);
   iVariables = variables;
Next, we proceed with the initialization of the newly declared objects. The first component to be initialized is the mask generation module.

It is worth noting that the framework's authors propose a mask generation module based on a trainable projection of the input data. In addition, a multi-head attention mechanism is used, with each attention head using its own unique mask.

Another important aspect is that attention is not applied across the entire frequency spectrum. The CATCH framework applies attention only within individual patches corresponding to the same frequency segment but belonging to different univariate sequences.

Naturally, the output of the mask generation module must be a probabilistic representation of the influence of individual channels in the form of real-valued coefficients.

To satisfy these requirements, our implementation employs a standard convolutional layer with a doubled kernel size (to accommodate complex-valued windows) and a number of filters equal to the dimensionality of the masking vector for a single sequence element.

   uint index = 0;
   if(!cMask.Init(0, index, OpenCL, 2 * iWindow, 2 * iWindow, iVariables * iHeads,
                                         iUnits, iVariables, optimization, iBatch))
      return false;
   cMask.SetActivationFunction(SIGMOID);
   CBufferFloat *temp = cMask.GetWeightsConv();
   if(!temp ||
      !temp.Fill(0))
      return false;
To ensure that the output values remain within the required range, we use a sigmoid activation function.

Note that during initialization, the trainable parameter matrix is filled with zeros. This allows all elements to begin with an equal influence mask value of 0.5. During training, the weights will be adjusted, gradually modifying the dependency mask between channels.

Next, we initialize the module responsible for generating the Query, Key, and Value representations. Here, we employ a complex-valued convolutional layer, since both the input and output of the module are expected to be complex-valued.

   index++;
   if(!cQKV.Init(0, index, OpenCL, iWindow, iWindow, 3 * iWindowKey * iHeads,
                                    iUnits, iVariables, optimization, iBatch))
      return false;
   cQKV.SetActivationFunction(None);
We then divide the combined tensor containing all three representations into two parts and extract the Query entity into a separate matrix. To support this operation, we initialize two additional objects. The method then completes execution, returning a Boolean result indicating whether all operations were performed successfully.

   index++;
   if(!cQ.Init(0, index, OpenCL, 2 * iWindowKey * iHeads * iVariables * iUnits, optimization, iBatch))
      return false;
   cQ.SetActivationFunction(None);
   index++;
   if(!cKV.Init(0, index, OpenCL, 2 * cQ.Neurons(), optimization, iBatch))
      return false;
   cKV.SetActivationFunction(None);
At the same stage, we initialize the object responsible for storing the multi-head attention results.

   index++;
   if(!cMHAttentionOut.Init(0, index, OpenCL, cQ.Neurons(), optimization, iBatch))
      return false;
   cMHAttentionOut.SetActivationFunction(None);
We then add a complex-valued convolutional layer for reducing the dimensionality of the attention outputs.

   index++;
   if(!cPooling.Init(0, index, OpenCL, iWindowKey * iHeads, iWindowKey * iHeads, 
                              iWindow, iUnits, iVariables, optimization, iBatch))
      return false;
   cPooling.SetActivationFunction(None);
Similar to the classical Transformer architecture, we add a layer to preserve residual connections.

   index++;
   if(!cResidual.Init(0, index, OpenCL, cPooling.Neurons(), optimization, iBatch))
      return false;
   cResidual.SetActivationFunction(None);
This is followed by two complex-valued convolutional layers of the FeedForward module.

   index++;
   if(!cFeedForward[0].Init(0, index, OpenCL, iWindow, iWindow, 4 * iWindow,
                                  iUnits, iVariables, optimization, iBatch))
      return false;
   cFeedForward[0].SetActivationFunction(LReLU);
   index++;
   if(!cFeedForward[1].Init(0, index, OpenCL, 4 * iWindow, 4 * iWindow, iWindow,
                                       iUnits, iVariables, optimization, iBatch))
      return false;
   cFeedForward[1].SetActivationFunction(None);
   SetActivationFunction(None);
//---
   return true;
  }
Finally, we complete the object initialization procedure, returning a Boolean result indicating the success of the operation to the calling program.

The next step is to construct the forward propagation algorithm within the feedForward method.

bool CNeuronComplexMVMHMaskAttention::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
The method receives a pointer to the input data object, whose validity is verified immediately.

To improve training stability, the input data is normalized.

   if(!NeuronOCL.SwapOutputs())
      return false;
   if(!SumAndNormilize(NeuronOCL.getPrevOutput(), NeuronOCL.getPrevOutput(), NeuronOCL.getOutput(),
                       iWindow, true, 0, 0, 0, 1))
      return false;
It should be noted that the input data consists of complex-valued quantities. Therefore, the summation and normalization methods were redefined to support complex-valued operations. The complete implementation can be found in the attached source code.

After normalization, the input data is used to generate the masking tensors and the Query, Key, and Value entities.

   if(!cMask.FeedForward(NeuronOCL))
      return false;
   if(!cQKV.FeedForward(NeuronOCL))
      return false;
We then split the combined tensor of generated entities into two separate tensors.

   if(!NeuronOCL.SwapOutputs())
      return false;
   if(!DeConcat(cQ.getOutput(), cKV.getOutput(), cQKV.getOutput(), 2 * iWindowKey * iHeads,
                                              4 * iWindowKey * iHeads, iUnits * iVariables))
      return false;
All prepared data is subsequently passed to the masked attention forward-pass kernel. This operation is performed within a dedicated method named AttentionOut.

   if(!AttentionOut())
      return false;
We then reduce the dimensionality of the resulting multi-head attention output and sum the output with the original input data to establish residual connections.

   if(!cPooling.FeedForward(cMHAttentionOut.AsObject()))
      return false;
   if(!SumAndNormilize(NeuronOCL.getOutput(), cPooling.getOutput(), cResidual.getOutput(), iWindow, true, 0, 0, 0, 1))
      return false;
The data is subsequently passed through the two convolutional layers of the FeedForward module.

   if(!cFeedForward[0].FeedForward(cResidual.AsObject()))
      return false;
   if(!cFeedForward[1].FeedForward(cFeedForward[0].AsObject()))
      return false;
Again we add residual connections. We save the final output in the inherited buffer used for data exchange with other neural layers of the model.

   if(!SumAndNormilize(cResidual.getOutput(), cFeedForward[1].getOutput(), getOutput(), iWindow, true, 0, 0, 0, 1))
      return false;
//---
   return true;
  }
The method concludes by returning the logical result of the operation to the calling program.

A few words should be said about the AttentionOut method, which is responsible for enqueuing the execution of the multi-head masked attention kernel MaskAttentionComplex. In recent work, we have rarely discussed the mechanics of submitting OpenCL kernels to the execution queue. This is understandable. The procedure for enqueuing kernels is fairly standard. And there is no need to repeatedly describe the same operations. However, in this case, there is an important nuance related to the CATCH framework.

As mentioned above, the authors of the CATCH framework propose performing masked attention strictly between elements within the same frequency range across different univariate sequences. In simple terms, we take identical frequency patches from different univariate sequences and analyze their interdependencies. This allows us to independently identify relationships within different univariate sequences across frequency bands—separately capturing long-term trends and short-term dynamics. However, this property was not explicitly accounted for when designing a unified masked attention kernel.

Nevertheless, we can still organize the required behavior by controlling the task space used for the kernel execution queue. To begin with, let us examine the dimensionality of the input data received by the object. We are working with segmented data from a multivariate sequence, which can be represented as a three-dimensional tensor: {Variable, Segment, Dimension}. Further, when constructing entities for multi-head attention, each output can be represented as a four-dimensional tensor: {Variable, Segment, Head, Dimension}.

It is important to note that our multi-head attention implementation processes each attention head independently. Therefore, if we merge the segment and head dimensions and treat each segment as an independent attention head, we effectively obtain an analysis of dependencies between identical segments across different univariate sequences. This is precisely the behavior required to correctly implement the CATCH framework.

The AttentionOut method does not take any parameters. In the method's control block, we only verify the validity of the pointer to the OpenCL context management object.

bool CNeuronComplexMVMHMaskAttention::AttentionOut(void)
  {
   if(!OpenCL)
      return false;
Next, we define arrays describing the kernel execution space. As discussed earlier, the sequence dimension is set to the number of univariate sequences in the dataset. For the number of heads, we use the product of the configured number of attention heads and the number of segments in each sequence.

   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iVariables/*Q units*/, iVariables/*K units*/, iHeads * iUnits/*Heads*/};
   uint local_work_size[3] = {1, iVariables, 1};
We then group execution threads into work-groups along the second dimension of the task space.

After that, we pass pointers to the required data buffers as kernel arguments.

   ResetLastError();
   int kernel = def_k_MaskAttentionComplex;
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_maskattcom_q, cQ.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_maskattcom_kv, cKV.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_maskattcom_scores, cMask.getPrevOutIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_maskattcom_out, cMHAttentionOut.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_maskattcom_masks, cMask.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
At this point, we take advantage of the fact that the attention coefficient matrix shares the same dimensionality as the channel masking matrix. This allows us to avoid allocating an additional buffer for temporary storage of attention coefficients. Instead, an unused buffer from the mask generation module is reused for this purpose.

We then pass the remaining necessary parameters to the kernel.

   if(!OpenCL.SetArgument(kernel, def_k_maskattcom_dimension, (int)iWindowKey))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(kernel, def_k_maskattcom_heads_kv, (int)(iHeads * iUnits)))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
It is also worth noting that, for both Key and Value representations, we again define the number of attention heads as the product of the user-defined head count and the sequence length.

Finally, we enqueue the kernel for execution.

   if(!OpenCL.Execute(kernel, 3, global_work_offset, global_work_size, local_work_size))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      return false;
     }
//---
   return true;
  }
The method ends by returning a logical result to the calling program.

After completing the forward-pass algorithms, we proceed to the construction of the backpropagation procedures. The first step is to implement the method responsible for distributing the error gradients across all internal components according to their contribution to the final model output: calcInputGradients.

bool CNeuronComplexMVMHMaskAttention::calcInputGradients(CNeuronBaseOCL *prevLayer)
  {
   if(!prevLayer)
      return false;
The method receives the same input data object used during the feed-forward pass. However, this time it is used to propagate the corresponding error gradients. We immediately verify the pointer validity.

At this stage, the external interface buffer of the object contains the error gradient received from the subsequent layer. These values have not yet been adjusted by the derivative of the activation function. This is intentional: the activation function was deliberately omitted during initialization. This is done to provide flexibility in selecting activation functions for internal objects. We now apply the derivative of the activation function of the final convolutional layer in the FeedForward module and propagate the corrected gradients accordingly. 

   if(!DeActivation(cFeedForward[1].getOutput(), cFeedForward[1].getGradient(), Gradient, cFeedForward[1].Activation()))
      return false;
We then propagate the gradients backward through the layers of the module up to the residual connection level following the attention block.

   if(!cFeedForward[0].calcHiddenGradients(cFeedForward[1].AsObject()))
      return false;
   if(!cResidual.calcHiddenGradients(cFeedForward[0].AsObject()))
      return false;
For the residual connection object, we also assume the absence of an activation function. We therefore repeat the gradient correction procedure using the derivative of the activation function applied to the attention output scaling layer.

   if(!DeActivation(cPooling.getOutput(), cPooling.getGradient(), cResidual.getGradient(), cPooling.Activation()) ||
      !DeActivation(cPooling.getOutput(), cPooling.getPrevOutput(), Gradient, cPooling.Activation()) ||
      !SumAndNormilize(cPooling.getGradient(), cPooling.getPrevOutput(), cPooling.getGradient(), iWindow,
                                                                                       false, 0, 0, 0, 1))
      return false;
However, it is important to emphasize that the residual connection object currently contains only the gradient flowing through the FeedForward path. We must also account for the gradient flowing through the residual attention path. Therefore, we again correct the gradient stored in the external interface buffer using the derivative of the activation function of the attention output projection layer.

It should be noted that reusing the external interface buffer does not imply double transformation of the data. In the first case, the corrected values were stored in the buffer of the final FeedForward layer, while the interface buffer itself retained the original gradient values. In the current step, the computations are written into the projection layer buffer without modifying the original data, which will later be reused when propagating gradients through the residual connection back to the input level.

After these corrections, we sum the contributions from both information flows. The gradient is then distributed across the attention heads.

   if(!cMHAttentionOut.calcHiddenGradients(cPooling.AsObject()))
      return false;
   if(!AttentionInsideGradients())
      return false;
We then call the AttentionInsideGradients method responsible for propagating gradients through the attention mechanism. We will not dwell on the details of this method here. We'll leave it for independent study. Its full implementation is provided in the attachment. However, it should be emphasized that the task space configuration and kernel scheduling parameters must be consistent with those used in the feed-forward pass.

At the next stage, we merge all gradient contributions from the individual attention components into a single tensor.

   if(!Concat(cQ.getGradient(), cKV.getGradient(), cQKV.getGradient(), 2 * iWindowKey * iHeads,
                                                  4 * iWindowKey * iHeads, iUnits * iVariables))
      return false;
We then propagate the error down to the input level, applying the derivative of the corresponding activation function.

   if(!DeActivation(cQKV.getOutput(), cQKV.getPrevOutput(), cQKV.getGradient(), cQKV.Activation()) ||
      !prevLayer.calcHiddenGradients(cQKV.AsObject()))
      return false;
However, this represents only one information flow. We also incorporate the gradient contribution from the residual connection path, applying the derivative of the activation function of the input data processing layer.

In this step, we use data both from the FeedForward module and from the external interface buffer.

   if(!DeActivation(prevLayer.getOutput(),cResidual.getPrevOutput(),cResidual.getGradient(),prevLayer.Activation()) ||
      !SumAndNormilize(prevLayer.getGradient(), cResidual.getPrevOutput(), cResidual.getPrevOutput(),
                       iWindow, false, 0, 0, 0, 1))
      return false;
   if(!DeActivation(prevLayer.getOutput(), cResidual.getGradient(), Gradient, prevLayer.Activation()) ||
      !SumAndNormilize(cResidual.getGradient(), cResidual.getPrevOutput(), cResidual.getPrevOutput(),
                       iWindow, false, 0, 0, 0, 1))
      return false;
It is important to recall that the input data was also used to generate the masking matrix. Therefore, we additionally include the gradient contribution associated with this computation path.

   if(!DeActivation(cMask.getOutput(), cMask.getGradient(), cMask.getGradient(), cMask.Activation()) ||
      !prevLayer.calcHiddenGradients(cMask.AsObject()) ||
      !SumAndNormilize(prevLayer.getGradient(), cResidual.getPrevOutput(), prevLayer.getGradient(),
                       iWindow, false, 0, 0, 0, 1))
      return false;
//---
   return true;
  }
At this point, we have accumulated the full gradient signal across all information flows at the input level, and we can conclude the method by returning a Boolean status to the calling program.

The parameter optimization method is left for independent study. Its logic is relatively straightforward, consisting primarily of calls to the corresponding methods of internal submodules. The full implementation is provided in the attachment, along with the complete code of the object and all its methods. We are moving on to the next stage of our work.


Assembling the CATCH Framework
It can be said that up to this point, a substantial amount of preparatory work has been completed, during which individual objects were developed. Now we proceed to integrate them into a unified structure of the CATCH framework. The execution logic of the framework is implemented within the CNeuronCATCH object, whose structure is shown below.

class CNeuronCATCH   :  public CNeuronTransposeOCL
  {
protected:
   CNeuronTransposeOCL        cTranspose;
   CNeuronBaseOCL             caFreqIn[2];
   CNeuronBaseOCL             cFreqConcat;
   CNeuronComplexConvOCL      caProjection[2];
   CNeuronComplexMVMHMaskAttention  caChannelFusion[2];
   CNeuronComplexConvOCL      caLinearHead[2];
   CNeuronBaseOCL             caFreqOut[2];
   //---
   virtual bool               FFT(CBufferFloat *inp_re, CBufferFloat *inp_im,
                                  CBufferFloat *out_re, CBufferFloat *out_im,
                                  uint variables, bool reverse = false);
   //---
   virtual bool               feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool               calcInputGradients(CNeuronBaseOCL *prevLayer) override;


public:
                     CNeuronCATCH(void) {};
                    ~CNeuronCATCH(void) {};
   //---
   virtual bool               Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint time_step, uint variables,
                                   uint window, uint step,
                                   uint window_key, uint heads,
                                   ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int                Type(void) override const   {  return defNeuronCATCH;   }
   //---
   virtual bool               Save(int const file_handle) override;
   virtual bool               Load(int const file_handle) override;
   //---
   virtual bool               WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void               SetOpenCL(COpenCLMy *obj) override;
  };
In the presented structure, we observe a relatively large number of internal components, each responsible for a specific part of the framework's complex processing algorithm. Their roles will be discussed progressively during the implementation of the class methods. For now, it is worth noting that all components are declared statically, which allows both the constructor and destructor to remain empty. The initialization of objects is performed in the Init method.

bool CNeuronCATCH::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                        uint time_step, uint variables,
                        uint window, uint step, uint window_key, uint heads,
                        ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronTransposeOCL::Init(numOutputs, myIndex, open_cl, variables, time_step, optimization_type, batch))
      return false;
The method receives a set of constants that uniquely define the architecture of the initialized object. Some of these parameters define the dimensionality of the input data. Others describe the structure of internal information flows. For example, time_step and variables define the number of time steps and univariate sequences in the multivariate time series. The window and step parameters define the frequency spectrum segmentation strategy. Meanwhile, window_key and heads represent the core parameters of the attention mechanism.

Within the method body, we first call the corresponding method of the parent class. In this case, the parent is a data transposition object, and this choice is intentional. The input is expected to be a multivariate time series represented as a matrix, where each row corresponds to the state vector of the environment at a given time step. However, the CATCH framework operates on the frequency-domain representation of individual univariate sequences. For convenience, we first transpose the input tensor and later restore it to its original representation. The latter operation is performed by the parent class, which is reflected in the parameters passed to it.

To transform the time series into the frequency domain, we use the Fast Fourier Transform. However, this algorithm requires the sequence length to be a power of two. To satisfy this constraint, we can extend any sequence by padding it with zeros, which does not affect the transform result. The next step is to determine the nearest valid sequence length.

//--- Calculate FFT size
   int power = int(MathLog(time_step) / M_LN2);
   if(power <= 0)
      return false;
   if(MathPow(2, power) != time_step)
      power++;
   uint FreqUinits = uint(MathPow(2, power));
We then compute the number of segments according to the specified segmentation parameters.

   if(window <= 0 || step <= 0)
      return false;
   int Segments = int((FreqUinits - int(window) + step - 1) / step);
   if(Segments <= 0)
      return false;
At this point, the preparatory stage is complete, and we proceed to initialize the internal components. The first is the transposition layer for the input data. Its initialization parameters are straightforward and require no special explanation.

   uint index = 0;
   if(!cTranspose.Init(0, index, OpenCL, time_step, variables, optimization, iBatch))
      return false;
Next, we create two data buffers to store the real and imaginary components of the frequency spectrum after Fourier decomposition.

   for(uint i = 0; i < caFreqIn.Size(); i++)
     {
      index++;
      if(!caFreqIn[i].Init(0, index, OpenCL, FreqUinits * variables, optimization, iBatch))
         return false;
      caFreqIn[i].SetActivationFunction(None);
     }
We then add a concatenation module for the frequency-domain representation.

   index++;
   if(!cFreqConcat.Init(0, index, OpenCL, 2 * FreqUinits * variables, optimization, iBatch))
      return false;
   cFreqConcat.SetActivationFunction(None);
For spectral segmentation, we use a complex-valued convolutional layer configured with the appropriate parameters.

   index++;
   if(!caProjection[0].Init(0, index, OpenCL, window, step, 2 * window, Segments, variables, optimization, iBatch))
      return false;
   caProjection[0].SetActivationFunction(LReLU);
A second convolutional layer completes the embedding construction for spectral patches.

   index++;
   if(!caProjection[1].Init(0, index, OpenCL,2*window,2*window,window_key, Segments, variables, optimization, iBatch))
      return false;
   caProjection[1].SetActivationFunction(TANH);
The analysis of interdependencies between frequency spectra of individual univariate sequences is performed using two consecutive attention modules. The first operates directly on the generated patch embeddings.

   index++;
   if(!caChannelFusion[0].Init(0, index, OpenCL,window_key,window_key,heads,Segments, variables, optimization, iBatch))
      return false;
The architecture of the second inter-channel masked attention layer depends on the number of segments. If the number of segments is divisible by two, pairwise merging of segments is performed to capture higher-order dependencies.

   index++;
   if(Segments % 2 == 0)
     {
      if(!caChannelFusion[1].Init(0, index, OpenCL, 2 * window_key, window_key, heads, Segments / 2,
                                                                   variables, optimization, iBatch))
         return false;
     }
   else
      if(!caChannelFusion[1].Init(0, index, OpenCL, window_key, window_key, heads, Segments, variables,
                                                                                 optimization, iBatch))
         return false;
Otherwise, the architecture remains identical to the previous attention layer.

The dimensionality of the tensor produced by the attention modules is then reduced back to the length of the original frequency spectrum using two consecutive projection convolution layers.

   index++;
   if(!caLinearHead[0].Init(0, index, OpenCL, window_key, window_key, window, Segments, variables,
                                                                            optimization, iBatch))
      return false;
   caLinearHead[0].SetActivationFunction(LReLU);
   index++;
   if(!caLinearHead[1].Init(0, index, OpenCL, window * Segments, window * Segments, FreqUinits, 
                                                               variables, 1, optimization, iBatch))
      return false;
   caLinearHead[



1 ].SetActivationFunction(None);
The first layer adjusts the segment dimensionality, while the second corrects the length of the univariate sequences.

We also introduce two additional components responsible for separating the real and imaginary parts of the frequency spectrum prior to applying the inverse Fourier transform.

   for(uint i = 0; i < caFreqOut.Size(); i++)
     {
      index++;
      if(!caFreqOut[i].Init(0, index, OpenCL, FreqUinits * variables, optimization, iBatch))
         return false;
      caFreqOut[i].SetActivationFunction(None);
     }
//---
   return true;
  }
At this stage, all internal components have been initialized, and the method concludes by returning a Boolean value indicating successful completion.

We now proceed to implement the forward pass in the feedForward method. As noted earlier, the method receives a pointer to the input data object containing the multivariate time series tensor.

bool CNeuronCATCH::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cTranspose.FeedForward(NeuronOCL))
      return false;
The input data is first transposed to facilitate processing of univariate sequences. It is then transformed into the frequency domain using the Fast Fourier Transform.

   if(!FFT(cTranspose.getOutput(), NULL, caFreqIn[0].getOutput(), caFreqIn[1].getOutput(), iCount, false))
      return false;
The resulting frequency representation is concatenated into a unified tensor.

   if(!Concat(caFreqIn[0].getOutput(), caFreqIn[1].getOutput(), cFreqConcat.getOutput(), 1, 1, caFreqIn[0].Neurons()))
      return false;
Next, spectral patching and embedding are performed using two projection convolution layers.

   CNeuronBaseOCL *neuron = cFreqConcat.AsObject();
   for(uint i = 0; i < caProjection.Size(); i++)
     {
      if(!caProjection[i].FeedForward(neuron))
         return false;
      neuron = caProjection[i].AsObject();
     }
Next, we analyze interdependencies using the inter-channel masked attention modules.

   for(uint i = 0; i < caChannelFusion.Size(); i++)
     {
      if(!caChannelFusion[i].FeedForward(neuron))
         return false;
      neuron = caChannelFusion[i].AsObject();
     }
The output is then projected back to the dimensionality of the original frequency spectrum.

   for(uint i = 0; i < caLinearHead.Size(); i++)
     {
      if(!caLinearHead[i].FeedForward(neuron))
         return false;
      neuron = caLinearHead[i].AsObject();
     }
We then split the frequency representation into real and imaginary components.

   if(!DeConcat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), neuron.getOutput(), 1, 1, 
                                                                     caFreqOut[0].Neurons()))
      return false;
   if(!FFT(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caFreqOut[0].getPrevOutput(),
                                                caFreqOut[1].getPrevOutput(), iCount, true))
      return false;
After that, the inverse Fourier transform is applied, returning the data from the frequency domain back into the time-domain sequence representation. However, it is important to note that the inverse Fast Fourier Transform produces a sequence whose length is a power of two. This may differ from the original sequence length; therefore, the excess values are discarded through tensor deconcatenation.

   if(!DeConcat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caFreqOut[0].getPrevOutput(), iWindow,
                                                       caFreqOut[0].Neurons() / iCount - iWindow, iCount))
      return false;
Residual connections from the transposed input data are then added to the resulting values, followed by normalization. Finally, the data is transformed back into the original time-series representation.

   if(!SumAndNormilize(caFreqOut[0].getOutput(),cTranspose.getOutput(),caFreqOut[0].getOutput(),iWindow,true,0,0,0,1))
     return false;
//---
   return CNeuronTransposeOCL::feedForward(caFreqOut[0].AsObject());
  }
This concludes the feed-forward computation, and the method returns a Boolean result to the calling program.

The next stage of development involves implementing the backward-pass procedures for the new object. Particular attention must be paid to the method responsible for distributing error gradients across all components according to their contribution to the final output: calcInputGradients.

bool CNeuronCATCH::calcInputGradients(CNeuronBaseOCL *prevLayer)
  {
   if(!prevLayer)
      return false;
The method receives a pointer to the input data object, which is immediately validated. The necessity of such validation has been discussed repeatedly earlier.

Next, the gradient received from the subsequent layer is transposed into the univariate sequence representation.

   if(!CNeuronTransposeOCL::calcInputGradients(caFreqOut[1].AsObject()))
      return false;
   if(!SumAndNormilize(caFreqOut[1].getGradient(),caFreqOut[1].getGradient(),cTranspose.getPrevOutput(),
                                                                              iWindow,false,0,0,0,0.5f))
     return false;
The resulting values are immediately copied into the available buffer of the input transposition object, representing the residual connection flow.

It is important to recall that the output of the inverse Fourier transform may exceed the expected sequence length. During the feed-forward pass, we discarded the excess values. It should be noted here that when performing the feed-forward pass, we padded the analyzed time series with zero values to the required size. Therefore, in the discarded part of the inverse Fourier Transform results, we expect to obtain similar zero values. Thus, for correct training behavior, the gradient corresponding to this discarded region is set to the previously obtained values with inverted sign.

   if((caFreqOut[0].Neurons() - iWindow) > 0)
      if(!SumAndNormilize(caFreqOut[1].getOutput(), caFreqOut[1].getOutput(), caFreqOut[1].getOutput(),
                                                                             1, false, 0, 0, 0, -0.5f))
         return false;
The gradients from both blocks are then concatenated into a single tensor.

   if(!Concat(caFreqOut[1].getGradient(), caFreqOut[1].getOutput(), caFreqOut[0].getGradient(), iWindow,
                                                              caFreqOut[0].Neurons() - iWindow, iCount))
      return false;
At this point, we obtain the gradient of the real part of the reconstructed signal. However, the imaginary part must also be considered. The inverse Fourier transform reconstructs a real-valued time series from its frequency representation. The time series itself is represented by real values whose imaginary part is zero. Therefore, the approach to defining the error gradient of imaginary part follows the same logic as the discarded real component, i.e., we change the sign of the previously obtained results.

   if(!SumAndNormilize(caFreqOut[1].getPrevOutput(), caFreqOut[1].getPrevOutput(), caFreqOut[1].getGradient(),
                                                                                    1, false, 0, 0, 0, -0.5f))
      return false;
The gradients are then transformed into the frequency domain using the Fast Fourier Transform.

   if(!FFT(caFreqOut[0].getGradient(), caFreqOut[1].getGradient(), caFreqOut[0].getOutput(), caFreqOut[1].getOutput(),
                                                                                                       iCount, false))
      return false;
The resulting values are concatenated into a unified tensor representing both real and imaginary components of the complex gradient.

   if(!Concat(caFreqOut[0].getOutput(), caFreqOut[1].getOutput(), caLinearHead[1].getGradient(), 1, 1,
                                                                              caFreqOut[0].Neurons()))
      return false;
Next, the gradient is propagated sequentially through all internal objects. First, through the final projection layers of the attention modules.

   if(!caLinearHead[0].calcHiddenGradients(caLinearHead[1].AsObject()))
      return false;
Then through the attention modules themselves, down to the embedding convolutional layers.

   CObject *neuron = caLinearHead[0].AsObject();
   for(int i = int(caChannelFusion.Size()) - 1; i >= 0; i--)
     {
      if(!caChannelFusion[i].calcHiddenGradients(neuron))
         return false;
      neuron = caChannelFusion[i].AsObject();
     }
This process continues until reaching the concatenated frequency-domain representation of the input data.

   for(int i = int(caProjection.Size()) - 1; i >= 0; i--)
     {
      if(!caProjection[i].calcHiddenGradients(neuron))
         return false;
      neuron = caProjection[i].AsObject();
     }
//---
   if(!cFreqConcat.calcHiddenGradients(neuron))
      return false;
At this stage, the result is split into real and imaginary components.

   if(!DeConcat(caFreqIn[0].getGradient(), caFreqIn[1].getGradient(), cFreqConcat.getGradient(), 1, 1,
                                                                               caFreqIn[0].Neurons()))
      return false;
The inverse Fourier transform is then applied to propagate the gradient back into the time-domain representation.

   if(!FFT(caFreqIn[0].getGradient(), caFreqIn[1].getGradient(), caFreqIn[0].getPrevOutput(),
                                                  caFreqIn[1].getPrevOutput(), iCount, false))
      return false;
Only the portion corresponding to the valid input data is retained.

   if(!DeConcat(cTranspose.getGradient(), caFreqIn[0].getGradient(), caFreqIn[0].getPrevOutput(), iWindow,
                                                        caFreqIn[0].Neurons() / iCount - iWindow, iCount))
      return false;
//---
   if(!SumAndNormilize(cTranspose.getGradient(),cTranspose.getPrevOutput(),cTranspose.getGradient(),iWindow,
                                                                                          false,0,0,0,1.0f))
     return false;
This result is then added to the previously stored residual connection gradients.

Finally, the gradients are transposed back into the original input representation and, if necessary, adjusted by the derivative of the activation function of the input layer.

   if(!prevLayer.calcHiddenGradients(cTranspose.AsObject()))
      return false;
   if(prevLayer.Activation() != None)
     {
      if(!DeActivation(prevLayer.getOutput(), prevLayer.getGradient(), prevLayer.getGradient(), prevLayer.Activation()))
         return false;
     }
//---
   return true;
  }
We return the logical result of the performed operations to the calling program and complete the execution of the method.

With this, we complete the presentation of our implementation of the approaches proposed in the CATCH framework. The full source code of all described components and methods is provided in the attachments.


Model Architecture
After examining the implementation algorithms of our proposed approach, we now briefly discuss the architecture of the trainable models. As in the previous work, we train three models: an environment state Encoder, an Actor, and a predictive model for the probability of the next directional movement. The CATCH framework is integrated into the encoder responsible for representing the observed state of the environment. Since the entire framework is encapsulated within a single object, the resulting model architecture is visually relatively simple.

bool CreateDescriptions(CArrayObj *&encoder, CArrayObj *&actor, CArrayObj *&probability)
  {
//---
   CLayerDescription *descr;
//---
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         return false;
     }
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
   if(!probability)
     {
      probability = new CArrayObj();
      if(!probability)
         return false;
     }
//--- Encoder
   encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (HistoryBars * BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
As usual, the first component is a fully connected layer used to encode the raw input data, followed by a batch normalization layer. This stage performs initial preprocessing of the multivariate time series and brings the univariate sequences into a comparable scale.

//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
Next, we apply our CATCH module. Here we use segmentation of the frequency spectrum into 8 elements with a step of 1. This allows for a more detailed analysis of interdependencies across the entire frequency spectrum of the input data. 

//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronCATCH;
   prev_count=descr.count = HistoryBars;
     {
      int temp[]={BarDescr,8,32,4};       // Variables, Frequency window, Key Size, Heads
      if(ArrayCopy(descr.windows, temp) < (int)temp.Size())
         return false;
     }
   descr.step=1;
   int prev_out=descr.windows[0];
   descr.batch = 1e4;
   descr.optimization=ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
The resulting representations are then rescaled into a narrow normalized range.

//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   descr.count = HistoryBars;
   descr.window = BarDescr;
   descr.step = BarDescr;
   descr.window_out = BarDescr;
   descr.layers = 1;
   descr.activation = TANH;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
Finally, we restore the original data distribution using an inverse normalization layer.

//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronRevInDenormOCL;
   descr.count = HistoryBars * BarDescr;
   descr.layers = 1;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
The architectures of the Actor and the probability forecasting model for the direction of the upcoming movement remain almost unchanged from the previous work. Readers are encouraged to review them independently. The full source code is provided in the attachment. The same applies to the environment interaction and model training scripts, which have been carried over without modification.


Testing
A substantial amount of work has been carried out to implement the CATCH framework approach in MQL5 and integrate it into trainable models. It is now time for the decisive stage — evaluating the effectiveness of the proposed solution on real historical data. This step allows us to assess both the strengths and weaknesses of the implementation and to determine its potential for further optimization.

For training, a dataset was constructed using random episodes generated in the MetaTrader 5 Strategy Tester. The dataset is based on historical EURUSD M1 data covering the entire year of 2024.

Model evaluation was performed on historical data from January to March 2025. All experimental parameters were kept unchanged to ensure objectivity of the results and to enable an unbiased assessment of strategy performance. This setup guarantees that the model does not merely memorize the training dataset but instead demonstrates its ability to adapt to new market conditions.

The testing results are presented below.

 

During the testing period, the model executed 68 trades, of which 32 were closed in profit, corresponding to a win rate of slightly above 47%. At the same time, the average profit per winning trade was nearly twice as large as the average loss per losing trade. As a result, the model achieved overall profitability over the test period, with a profit factor of 1.72.


Conclusion
Across these two articles, we explored the theoretical foundations of the CATCH framework, an innovative approach that combines Fourier transform techniques with frequency-domain pat

READ 3
Yevgeniy Koshtenko
Prices play a melody that repeats on certain days of the month, days of the week, or even hours of the day. These recurring rhythms, or seasonal patterns, can provide a trader with clues as to when the market is likely to rise and when it is likely to fall. Seasonality in financial markets is not just a curious phenomenon, but a tool that helps identify predictable moments in price chaos. For example, have you noticed that some currency pairs often rise on Mondays or fall at the end of the month? This is seasonality, and studying it can give traders an edge.

In this guide, we will create a seasonality indicator in MQL5 for the MetaTrader 5 platform. Our indicator will analyze historical price data to identify the average return for days of the month (1st to 31st), days of the week (Monday to Sunday), or hours of the day (0 to 23). The results will be displayed as a histogram in a separate chart window, with a forecast line connecting the seasonality values and dots highlighting the expected values on the following bars. In addition, the indicator will display text statistics with forecasts, best and worst periods. We will walk you through the development process step by step, explaining every part of the code so you can not only use the indicator but also adapt it to your own ideas. This guide takes you through both programming and financial analysis, where code becomes the bridge between data and trading decisions.


Why study seasonality?
Seasonality is like the hidden pulse of the market. It arises from many factors: trader behavior, economic events, even human psychology. For example, at the end of the month, large funds may close positions, causing predictable price movements. Or at certain times of the day, market activity increases due to the opening of trading sessions in London or New York. Our indicator will help you identify these patterns by turning chaotic data into clear charts and statistics. This is not magic but data analysis working in the trader’s favor.

The indicator we will create will be universal. It allows users to select the type of seasonality, set the number of bars to analyze, display returns as a percentage or absolute value, and enable or disable the forecast on the chart. Flexibility is the key to making the indicator a useful tool for different markets and timeframes, whether it is the daily EURUSD chart or the hourly gold chart.


Defining the structure of the indicator
Creating an indicator begins with understanding how it will look and work. We want it to be displayed in a separate window below the price chart so as not to clutter the main space. We use three elements for visualization: a histogram that shows the average return for each period, a forecast line that connects the seasonality values and the forecast data, and forecast points that highlight the expected values on the next two bars. Each element will be associated with a data buffer - an array that stores the values to be drawn. We will also need global variables to store seasonality data and period labels, as well as inputs so that the user can customize the indicator.

//+------------------------------------------------------------------+
//|                                         SeasonalityIndicator.mq5 |
//|                                  Copyright 2025, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

//--- plot Seasonality
#property indicator_label1  "Seasonality"
#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot Forecast Line
#property indicator_label2  "Forecast"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrRed
#property indicator_style2  STYLE_DASH
#property indicator_width2  2

//--- plot Forecast Points
#property indicator_label3  "Forecast Points"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrOrange
#property indicator_width3  3
In this code, we set basic information about the indicator: copyright, link to the MetaQuotes website, and version. The #property indicator_separate_window directive specifies that the indicator will be in a separate window. We define three buffers and three plots. The DRAW_HISTOGRAM histogram is colored blue, the DRAW_LINE forecast line is colored red, and the DRAW_ARROW forecast points are colored orange with a width of three pixels. These settings define the visual style that will make the indicator clear and attractive.


Set up inputs
To make the indicator flexible, we add inputs that the user can change in the MetaTrader 5 settings window. The parameter for selecting the seasonality type allows us to analyze days of the month, days of the week, or hours of the day. The number of bars to analyze determines how deep into history we can look. It is possible to display returns as a percentage or absolute value, as well as to show only positive seasonality values. The option to enable the forecast on the chart and customize the colors of the line and forecast points makes the indicator even more customizable.

//--- Input parameters
enum ENUM_SEASONALITY_TYPE
{
   SEASONALITY_DAYS_OF_MONTH = 0,  // Days of the month (1-31)
   SEASONALITY_DAYS_OF_WEEK = 1,   // Days of the week (Mon-Sun)
   SEASONALITY_HOURS = 2           // Hours (0-23)
};

input ENUM_SEASONALITY_TYPE SeasonalityType = SEASONALITY_DAYS_OF_MONTH; // Seasonality type
input int BarsToAnalyze = 1000;                                          // Number of bars to analyze 
input bool ShowPercentage = true;                                        // Show as a percentage
input bool ShowPositiveOnly = false;                                     // Show only positive values
input bool ShowForecastOnChart = true;                                   // Show forecast on chart
input color ForecastColor = clrRed;                                      // Forecast line color
input color ForecastPointsColor = clrOrange;                             // Color of forecast points
The ENUM_SEASONALITY_TYPE enumeration defines three seasonality options. By default, analysis by days of the month is selected. The BarsToAnalyze parameter is set to 1000 bars, which corresponds to approximately four years of data on a daily chart. The ShowPercentage option is enabled by default to display returns as a percentage, which is more intuitive. The ShowPositiveOnly and ShowForecastOnChart parameters provide additional flexibility, and the colors can be changed through the MetaTrader interface.


Defining buffers and variables
To store data, the indicator uses three buffers: one for the histogram, the second for the forecast line, and the third for forecast points. Global variables are also needed: an array for storing the average return by period, an array for period labels (for example, "Mon" or "1"), and a variable for the number of periods, which depends on the type of seasonality (31 for days of the month, 7 for days of the week, 24 for hours).

//--- Indicator buffers
double SeasonalityBuffer[];
double ForecastLineBuffer[];
double ForecastPointsBuffer[];

//--- Global variables
double seasonality_data[];
string period_labels[];
int periods_count;
The buffers will contain the data to be rendered, and the seasonality_data and period_labels variables will help store and display the analysis results. The periods_count variable determines how many periods are analyzed and is set depending on the type of seasonality.


Indicator initialization
The OnInit function is executed when the indicator starts and is responsible for the initial setup. It binds the buffers to the indicator plots, sets the arrow style for forecast points, and sets colors from user parameters. Depending on the seasonality type, labels are created for the periods, and the array for storing seasonality data is initialized to zeros. The display precision is adjusted depending on whether percentages or absolute values are selected.

int OnInit()
{
   SetIndexBuffer(0, SeasonalityBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ForecastLineBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, ForecastPointsBuffer, INDICATOR_DATA);
   
   PlotIndexSetInteger(2, PLOT_ARROW, 159);
   
   PlotIndexSetInteger(1, PLOT_LINE_COLOR, ForecastColor);
   PlotIndexSetInteger(2, PLOT_LINE_COLOR, ForecastPointsColor);
   
   switch(SeasonalityType)
   {
      case SEASONALITY_DAYS_OF_MONTH:
         periods_count = 31;
         ArrayResize(period_labels, periods_count);
         for(int i = 0; i < periods_count; i++)
            period_labels[i] = IntegerToString(i + 1);
         IndicatorSetString(INDICATOR_SHORTNAME, "Seasonality by day of the month");
         break;
         
      case SEASONALITY_DAYS_OF_WEEK:
         periods_count = 7;
         ArrayResize(period_labels, periods_count);
         period_labels[0] = "Mon";
         period_labels[1] = "Tue";
         period_labels[2] = "Wed";
         period_labels[3] = "Thu";
         period_labels[4] = "Fri";
         period_labels[5] = "Sat";
         period_labels[6] = "Sun";
         IndicatorSetString(INDICATOR_SHORTNAME, "Seasonality by day of the week");
         break;
         
      case SEASONALITY_HOURS:
         periods_count = 24;
         ArrayResize(period_labels, periods_count);
         for(int i = 0; i < periods_count; i++)
            period_labels[i] = IntegerToString(i) + ":00";
         IndicatorSetString(INDICATOR_SHORTNAME, "Seasonality by hours");
         break;
   }
   
   ArrayResize(seasonality_data, periods_count);
   ArrayInitialize(seasonality_data, 0.0);
   
   IndicatorSetInteger(INDICATOR_DIGITS, ShowPercentage ? 2 : _Digits);
   
   return(INIT_SUCCEEDED);
}
This function sets the basis for the indicator operation. For example, for days of the week, an array of labels is created from "Mon" to "Sun", and for hours - from "0:00" to "23:00". The indicator name changes so that the user immediately understands which type of seasonality is selected. This makes the indicator intuitive.


Calculation of seasonality
The main calculations are performed in the OnCalculate function. It analyzes historical price data, calculates the average return for each period, and fills the buffers for rendering. First, we check whether there are enough bars on the chart. Then auxiliary arrays are created to sum the returns and count the bars in each period.

For each bar, the return is calculated as the ratio of the difference in closing prices to the current closing price. The GetPeriodIndex function determines which period the bar belongs to, and the data is written to arrays. After this, the average return is calculated, which, if necessary, is converted into a percentage or reset to zero for negative values. The buffers are filled with data, and a separate function is called for the forecast.

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   if(rates_total < BarsToAnalyze + 1)
      return(0);

   ArrayInitialize(seasonality_data, 0.0);
   
   double period_returns[];
   int period_counts[];
   ArrayResize(period_returns, periods_count);
   ArrayResize(period_counts, periods_count);
   ArrayInitialize(period_returns, 0.0);
   ArrayInitialize(period_counts, 0);
   
   int start_pos = MathMax(0, rates_total - BarsToAnalyze - 1);
   
   for(int i = start_pos; i < rates_total - 1; i++)
   {
      double return_value = 0.0;
      if(close[i] != 0)
         return_value = (close[i+1] - close[i]) / close[i];
      
      int period_index = GetPeriodIndex(time[i]);
      
      if(period_index >= 0 && period_index < periods_count)
      {
         period_returns[period_index] += return_value;
         period_counts[period_index]++;
      }
   }
   
   for(int i = 0; i < periods_count; i++)
   {
      if(period_counts[i] > 0)
      {
         seasonality_data[i] = period_returns[i] / period_counts[i];
         if(ShowPercentage)
            seasonality_data[i] *= 100.0;
         if(ShowPositiveOnly && seasonality_data[i] < 0)
            seasonality_data[i] = 0.0;
      }
   }
   
   for(int i = 0; i < rates_total; i++)
   {
      int period_index = GetPeriodIndex(time[i]);
      if(period_index >= 0 && period_index < periods_count)
      {
         SeasonalityBuffer[i] = seasonality_data[period_index];
         ForecastLineBuffer[i] = seasonality_data[period_index];
      }
      else
      {
         SeasonalityBuffer[i] = 0.0;
         ForecastLineBuffer[i] = 0.0;
      }
      ForecastPointsBuffer[i] = EMPTY_VALUE;
   }
   
   if(ShowForecastOnChart)
      DrawForecastOnChart(rates_total, time);
   
   return(rates_total);
}
This code is the heart of the indicator. It turns raw price data into meaningful patterns. Imagine you are analyzing a daily chart of a currency pair. For each day of the week, the indicator calculates how much the price has changed and finds the average value. If you see that the market falls by an average of 0.05% on Fridays, this could be a signal for a trading strategy.


Definition of the period
To understand which period each bar belongs to, we create the GetPeriodIndex function. It converts bar time into a period index depending on the type of seasonality. For days of the month, it takes the day number and subtracts one to get an index from 0 to 30. For days of the week, it converts the weekday value (where Sunday is 0 and Monday is 1) to a day of the week order where Monday is 0 and Sunday is 6. For hours, the hour value from 0 to 23 is simply returned.

int GetPeriodIndex(datetime bar_time)
{
   MqlDateTime dt;
   TimeToStruct(bar_time, dt);
   
   switch(SeasonalityType)
   {
      case SEASONALITY_DAYS_OF_MONTH:
         return(dt.day - 1);
         
      case SEASONALITY_DAYS_OF_WEEK:
         return(dt.day_of_week == 0 ? 6 : dt.day_of_week - 1);
         
      case SEASONALITY_HOURS:
         return(dt.hour);
   }
   
   return(-1);
}
This function is like a compass that helps the indicator understand which "box" to assign the bar data to. Without it, we would not be able to associate the price with a specific day or hour.


Drawing the forecast
The forecast is what makes the indicator not only analytical, but also practically useful. The DrawForecastOnChart function draws the expected seasonality values on the last two bars of the chart. It determines the current period and the next two periods using the GetNextPeriod function. Seasonality values for these periods are written to the forecast line and point buffers. To make the forecast line look smooth, the value of the current period is written to the buffer for the penultimate bar.

void DrawForecastOnChart(int rates_total, const datetime &time[])
{
   if(rates_total < 3)
      return;
   
   datetime current_time = time[rates_total - 1];
   int current_period = GetPeriodIndex(current_time);
   int next_period1 = GetNextPeriod(current_period);
   int next_period2 = GetNextPeriod(next_period1);
   
   int forecast_pos1 = rates_total - 2;
   int forecast_pos2 = rates_total - 1;
   
   if(next_period1 >= 0 && next_period1 < periods_count)
   {
      ForecastLineBuffer[forecast_pos1] = seasonality_data[next_period1];
      ForecastPointsBuffer[forecast_pos1] = seasonality_data[next_period1];
   }
   
   if(next_period2 >= 0 && next_period2 < periods_count)
   {
      ForecastLineBuffer[forecast_pos2] = seasonality_data[next_period2];
      ForecastPointsBuffer[forecast_pos2] = seasonality_data[next_period2];
   }
   
   if(rates_total >= 3)
   {
      int current_pos = rates_total - 3;
      int current_period_idx = GetPeriodIndex(time[current_pos]);
      
      if(current_period_idx >= 0 && current_period_idx < periods_count)
      {
         double current_value = seasonality_data[current_period_idx];
         double next_value = (next_period1 >= 0) ? seasonality_data[next_period1] : current_value;
         
         ForecastLineBuffer[current_pos] = current_value;
         if(forecast_pos1 < rates_total)
            ForecastLineBuffer[forecast_pos1] = next_value;
      }
   }
}
A forecast is like a look into the future: the indicator tells you what to expect based on historical data. If, for example, Tuesdays typically return +0.07%, the indicator will show this on the chart, helping the trader prepare.


Calculating the next period
The GetNextPeriod function determines which period follows the current one. It uses modular arithmetic to provide a cyclic transition: after the 31st day of the month comes the 1st, after Sunday comes Monday, after 23:00 comes 0:00.

int GetNextPeriod(int current_period)
{
   if(current_period < 0)
      return(-1);
   
   switch(SeasonalityType)
   {
      case SEASONALITY_DAYS_OF_MONTH:
         return((current_period + 1) % 31);
         
      case SEASONALITY_DAYS_OF_WEEK:
         return((current_period + 1) % 7);
         
      case SEASONALITY_HOURS:
         return((current_period + 1)rate_total % 24);
   }
   
   return(-1);
}
This function ensures that the forecast always looks at the correct future periods, preserving the cyclical time logic. It acts as a precise market calendar, ensuring that the indicator moves correctly from one period to the next, whether it be the changing of the days of the month, days of the week, or hours. Think of the market as a giant clock, where every tick counts not only time but also potential price movements. After the 31st day of the month, the function smoothly returns us to the 1st day, after Sunday it moves to Monday, and after 23:00 - to 0:00. This cyclical approach reflects the real time structure of financial markets, allowing the indicator to predict which periods may be significant for a trader.


Displaying statistics


To enable traders to instantly understand the analysis results, the indicator displays detailed statistics directly on the chart. The ShowStatistics function creates a text message that, like a storyteller, turns dry numbers into a fascinating story of market patterns.

It begins with a title that changes depending on the type of seasonality chosen — whether it is analysis by day of the month, day of the week, or hour of the day. The number of bars analyzed is then indicated, giving the trader an idea of the depth of historical data underlying the calculations. Next, the forecast obtained from the GetForecast function is added, which predicts how the market might perform over the next two periods based on historical trends. After this, the function carefully examines the seasonality_data array, finding the periods with the highest and lowest returns. These periods are labeled as best and worst, and their values are displayed to three decimal places so that the trader can assess their significance.

The message concludes with full statistics, where for each period its label is shown — for example, "Mon" or "15" — and the average return. The text appears in the upper-right corner of the chart as a compact analytical summary of market rhythms, making them understandable and useful for making trading decisions.


Conclusion
Seasonality in financial markets is not a mystical phenomenon, but a mathematically proven pattern that can become a powerful tool in a trader's arsenal. The indicator we created transforms chaotic price movements into understandable patterns, revealing hidden market rhythms by day of the month, day of the week, or trading session hour.

With flexible parameter settings, visual display of results in the form of a histogram with a forecast, and detailed statistics, this tool allows traders to not only analyze historical data but also make informed assumptions about future price movements. The integration of forecast elements makes the indicator practically applicable for developing trading strategies.

The indicator code is easily adapted to various assets and timeframes, and a step-by-step explanation of each function allows us to modify the algorithm to suit our own trading ideas. Ultimately, this indicator is the bridge between complex statistical calculations and concrete trading decisions, turning data into knowledge and knowledge into potential profit.



READ 5
Backtracking Search Algorithm (BSA)
MetaTrader 5 — Trading systems | 4 June 2026 at 14:18
1 038
0
Andrey Dik
Andrey Dik

Contents
Introduction
Implementation of the algorithm
Test results


Introduction
In the endless labyrinth of possibilities, where every turn can lead to either triumph or a dead end, the wise traveler leaves behind invisible traces —something ephemeral, yet more reliable: the memory of the paths traveled. This idea (looking back to see the future) lies at the heart of the optimization algorithm. Every step into the unknown is taken with an eye on past experience, where history becomes a compass, and memory becomes a map.

In this article, I will consider the algorithm that I found very interesting due to its search concept. The Backtracking Search Algorithm (BSA) is a new evolutionary algorithm (EA) for solving real-valued numerical optimization problems, proposed by Pinar Civicioglu in 2013. It is a method of finding the best solution that can "learn from past experience". 



Implementation of the algorithm
BSA works on the principle of evolutionary algorithms, but has unique features, using two populations:
current population (P) - an actively evolving population
historical population (oldP) - a randomly selected population from past generations
BSA uses a random mutation strategy that produces only one directional solution for each target individual. Mutation formula:

Mutant = P + F × (oldP - P)
where F is the amplitude factor that controls the search step size.

The BSA crossover process involves two steps. The first strategy uses "mixrate". The second strategy allows only one randomly selected coordinate to be altered in each trial vector. Imagine that you and your friends (that is our population) are looking for the best pizzeria in town. 

Initial situation:

you have 10 friends (population size),
everyone starts searching in a random area of the city,
everyone has a smartphone with a map of past trips (historical memory).
Day 1: Everyone went out into the city. The first friend found a pizzeria with a 7/10 rating, the second friend found a pizzeria with a 5/10 rating, the third discovered a pizzeria with an 8/10 rating...and so on.

Day 2: Using the accumulated experience.

Step 1: "Remembering the past" (Selection-I). Algorithm: "Let's flip a coin!" If it lands on heads, then "Remember where everyone was yesterday" (update history), if it lands on tails, "Use old memories" (do not update), then "Shuffle memories" (like a deck of cards).

Step 2: "Follow the trail" (Mutation). Every friend thinks: "Where am I now? (current position) and where was I before? (historical position). Then the formula for movement will be: New place = Current place + Random step × (Old place - Current place). For example: the first friend is now on A street, 10, his “ghost from the past” was on B street, 20, random step = 2. New location = A.,10 + 2 × (B.,20 - A.,10) = movement towards B street, but 2 times further!

"Correcting the route" (Crossover). The algorithm tosses a coin and chooses a strategy. Strategy A: "Partial change". The first friend planned to go to a certain address, but the algorithm says: "Change only the street, leave the house as it was". Result: okay, we change the street, but keep the house. Strategy B: "Minimal change". The first friend planned to go to a certain address, but the algorithm says: "Stay at the old place, but change only the house number." Result: ok, we change the house number.

"Check the overall result" (Selection-II). Friend 1 came to a new place:

If the new pizzeria is better (rating 9/10): "Great, I'm staying here!"
If worse (rating 4/10): "No, I'm coming back!"
The figure below illustrates how the algorithm works.

bsa

Figure 1. BSA algorithm phases

This illustration shows how the BSA algorithm searches for an optimal solution in a two-dimensional space. Imagine you are looking at a topographic map from above, where the goal is to find the highest point (red center).

The illustration is divided into three panels showing the evolution of the search: Iteration 1 – Random Initialization. The square search field with contour lines (like a topographic map) has a red dot in the center representing the global optimum (the best solution), and four blue dots (P1, P2, P3, P4) are the initial random positions of the "explorers". The algorithm randomly places four agents throughout the search space. At this stage, oldP = P (the historical population copies the current one), and this is the starting point of the search.

Iteration 2 - Mutation Step. Blue dots are the current positions of agents, green translucent dots are positions from historical memory (mixed), red arrows are the directions of movement during mutation.

Key elements: The red arrows show the mutation formula in action: M = P + F × (oldP - P). Each agent moves relative to its "historical counterpart". Some arrows point towards historical positions, others away from them (depending on the F sign). Formula in red box: F = 3 × randn(); M = P + F × (oldP - P). This is the key formula for the BSA mutation.

Iteration 3 - After Crossover & Selection. Purple dots are new positions after the crossover (Trial population), translucent blue dots are previous positions (for comparison), and green arrows show only improvements (where the new position is better than the old one). Crossover: The algorithm combined information from the mutant and current populations. Greedy selection kept only those moves that improved the solution. The population has approached the optimum (red center).

BSA Process Summary, the full algorithm cycle as a sequence of colored circles:
Init (blue) - random start
Select-I (green) - memory update with a 50% probability
Mutate (red) - application of the mutation formula
Crossover (purple) – combination of solutions
Evaluate (orange) - fitness calculation
Select-II (turquoise) - greedy selection of the best
The dotted arrow shows that the process is repeated until convergence. This image clearly demonstrates why BSA is effective: it "remembers" where it has been before and uses that information to search more intelligently than a simple random walk.

We can now move on to the BSA pseudocode.

Preparation for work
Initial parameters:

Set the population size 
Set the mixrate crossover parameter 
Create empty containers for:
current population
historical population
mutant population
trial population
crossover maps for each individual
Initialization:

Place all individuals of the historical population randomly in the search space
Take into account discretization if the step is given
Set the "selection required" flag to "no"
The main loop of the algorithm
STEP 1: First launch

If this is the first iteration:

Place the current population randomly
Apply discretization to coordinates
Mark that initialization is complete
Exit and wait for fitness to be calculated
STEP 2: Greedy selection (if required)

If the "selection required" flag is set:

For each individual, compare:
if the new solution is worse than the saved one, we return the old coordinates and fitness
if it is better, we leave the new one.
Reset selection flag
STEP 3: Saving the current state

For each individual:

Save the current fitness
Save the copy of the current coordinates
STEP 4: Updating historical memory (Selection-I)

Toss a coin (50% chance)
If the coin lands on heads:
copy the entire current population into historical memory
preserve their fitness
Shuffle the historical population like a deck of cards:
go from the last individual to the first
for each, select a random position for exchange
swap
STEP 5: Mutation

Generate a motion factor F from a normal distribution
average = 0, range from -3 to +3
like a dice roll, but with a bell-shaped distribution
For each individual and each coordinate:
New position = Current + F × (Historical - Current)
if F is positive → movement to the historical position
if F is negative → movement from the historical position
STEP 6: Crossover

Copy the mutant population into the trial population
Toss a weighted coin (40% / 60%)
If the 40% branch is selected (Strategy 1): 

For each individual:

determine how many coordinates to change (from 0 to all, using mixrate)
randomly select the coordinates
for the selected coordinates, take values from the current population
leave the rest from the mutant one
If 60% is rolled (Strategy 2):

For each individual:

choose only one random coordinate
replace it with a value from the current population
leave all the rest from the mutant one
STEP 7: Boundary control

For each individual of the trial population:

Check each coordinate
If we go beyond the boundaries:
toss a coin (50/50)
if "heads" → generate a new random value within the bounds
if "tails" → place on the nearest border
Apply discretization if step is specified
STEP 8: Prepare for the assessment

Copy the trial population to the main population for evaluation
Set the "selection required" flag to "yes"
Pass control to fitness calculation
After calculating the fitness 
Find the individual with the best fitness
If you find something better than the global record:
update the global record
save the coordinates of the best solution
Repetition
Return to STEP 2 and continue until:

convergence is reached
or the iteration limit is exceeded
Now that we understand the main points, let's start writing the algorithm code. The C_AO_BSA_Backtracking class will be an implementation of the BSA backtracking algorithm for solving optimization problems. It will inherit the C_AO base class, which defines a common interface for optimization algorithms. 

Main characteristics and purpose:

Optimization parameters:
popSize - the population size, the number of "agents" or "solutions" that the algorithm considers simultaneously. 
mixrate - a crossover parameter that controls the degree of "mixing" of information between agents when creating new solutions. 
Initialization:
The class constructor initializes the base parameters and adds popSize and mixrate to the params array, which is used to control the algorithm parameters.
The SetParams method allows you to update the internal parameters of the algorithm based on the values obtained from the params array. It also includes basic checks for value validity.
Algorithm life cycle (methods):
Init — for the initial setup of the algorithm, including setting the ranges of change of variables (minimum, maximum values and steps) and the number of optimization epochs.
Moving — describes the main iteration of the algorithm, responsible for generating new "candidates" or "moving" solutions based on the current population.
Revision — evaluates the generated solutions and updates the population based on their quality.
Internal data structures for the algorithm:
oldP - an array representing the "historical" or previous population of agents.
M - an array for the population resulting from mutation.
T - an array for the "trial" population, used for comparison with the current population before making decisions.
F — amplitude factor for mutation, which influences the magnitude of change upon mutation.
needSelection — flag indicating the need to perform the second stage of selection.
prevFitness — array for storing the fitness values of agents from the previous iteration.
S_Map — auxiliary structure containing a binary map, used during the crossover process to determine, which agent variables will be "mixed" from different parents.
map — an array of these binary maps, one for each agent.
The main steps of the algorithm (internal methods):
SelectionI — the first stage of selection, responsible for choosing agents to create new solutions.
Mutation — apply the mutation operation to the selected agents.
Crossover — perform a crossover (mixing) operation between agents to create new "test" solutions.
BoundaryControl — check that the agent parameter values remain within the specified limits (minimum and maximum).
ShufflePopulation — method for randomly mixing a population.
Thus, the C_AO_BSA_Backtracking class implements an evolutionary optimization algorithm that uses the concepts of population, mutation, and crossover, as well as BSA-specific mechanisms such as backtracking, to solve optimization problems.

//————————————————————————————————————————————————————————————————————
class C_AO_BSA_Backtracking : public C_AO
{
  public: //----------------------------------------------------------
  ~C_AO_BSA_Backtracking () { }
  C_AO_BSA_Backtracking ()
  {
    ao_name = "BSA";
    ao_desc = "Backtracking Search Algorithm";
    ao_link = "https://www.mql5.com/en/articles/18568";

    popSize = 10;     // population size
    mixrate = 1.0;    // crossover parameter

    ArrayResize (params, 2);

    params [0].name = "popSize"; params [0].val = popSize;
    params [1].name = "mixrate"; params [1].val = mixrate;
  }

  void SetParams ()
  {
    popSize = (int)params [0].val;
    mixrate = params      [1].val;

    // Check the parameters validity
    //if (popSize < 2) popSize = 2;
    if (mixrate < 0.0) mixrate = 0.0;
    if (mixrate > 1.0) mixrate = 1.0;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //------------------------------------------------------------------
  double mixrate;        // crossover parameter

  private: //---------------------------------------------------------
  S_AO_Agent oldP [];    // historical population
  S_AO_Agent M    [];    // mutant population (Mutant)
  S_AO_Agent T    [];    // trial population (Trial)

  double F;              // amplitude factor for mutation
  bool   needSelection;  // flag for the necessity of executing Selection-II
  double prevFitness []; // array for storing previous fitness

  // Auxiliary structures for crossover
  struct S_Map
  {
      int val [];      // binary map for crossover

      void Init (int size)
      {
        ArrayResize (val, size);
        ArrayInitialize (val, 0);
      }
  };

  S_Map map [];        // array of binary maps for each agent

  // Algorithm methods
  void SelectionI        ();
  void Mutation          ();
  void Crossover         ();
  void BoundaryControl   (S_AO_Agent &agent);
  void ShufflePopulation (S_AO_Agent &pop []);
};
//————————————————————————————————————————————————————————————————————
The Init method is responsible for preparing the algorithm for operation before starting optimization. First of all, the basic initialization function is called, which sets the basic parameters related to the ranges of variable values (minimum, maximum values and step of change). If this basic step fails, the method aborts and returns failure.

Next, the internal data structures specific to the BSA algorithm are allocated and prepared. In particular, population arrays are created and brought to the required size: the historical population oldP, the mutant population M, the trial population T, as well as an array of binary maps for crossover "map" and the prevFitness array for storing the previous fitness values of each agent. The flag for the need for additional selection is initially set to 'false'.

After this, initialization methods are called for each agent in the population, which prepare the internal structures of each agent in accordance with the number of task parameters.

Then the historical population is filled with initial values. For each agent and each parameter in its set, a random value is generated within a given range, after which this value is adjusted according to a given change step. If all these steps are successfully completed, the method returns a value indicating that the algorithm object has been successfully initialized and is ready for further optimization.

//————————————————————————————————————————————————————————————————————
//--- Initialization
bool C_AO_BSA_Backtracking::Init (const double &rangeMinP  [],
                                  const double &rangeMaxP  [],
                                  const double &rangeStepP [],
                                  const int epochsP = 0)
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //------------------------------------------------------------------
  // Initialize additional BSA structures
  ArrayResize (oldP, popSize);
  ArrayResize (M,    popSize);
  ArrayResize (T,    popSize);
  ArrayResize (map,  popSize);
  ArrayResize (prevFitness, popSize);

  needSelection = false;

  for (int i = 0; i < popSize; i++)
  {
    oldP [i].Init (coords);
    M    [i].Init (coords);
    T    [i].Init (coords);
    map  [i].Init (coords);
  }

  // Initialize oldP historical population
  for (int p = 0; p < popSize; p++)
  {
    for (int c = 0; c < coords; c++)
    {
      oldP [p].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
      oldP [p].c [c] = u.SeInDiSp (oldP [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }

  return true;
}
//————————————————————————————————————————————————————————————————————
The Moving method describes the main step of the BSA optimization algorithm and is responsible for generating new solution candidates in the population. On the first call of the method, when the "revision" flag is 'false', the active population "a" is initially populated. For each agent in the population and for each of its parameters, a random value is generated within a given range (minimum and maximum). This random value is then adjusted according to the specified parameter change step. After initialization, the "revision" flag is set to 'true' (so that this block is not executed on subsequent calls), and the needSelection flag is cleared. The method terminates.

Implementing greedy selection (Selection-II) (performed if required): if needSelection is 'true', it means that a new population was generated in the previous step, and now its fitness needs to be compared with the fitness of the previous solutions. Iteration occurs for each agent in the population. The current fitness value of the "a[i].f" agent (which was calculated for the "trial" solution) is compared with its fitness at the previous step stored in prevFitness[i]. If the "trial" solution a [i] turned out to be worse than the previous one, then the coordinate values of the current "a [i].c" agent are restored from "a [i].cP" (the coordinates of the agent at the previous step), and its "a [i].f" fitness is also restored to prevFitness [i]. This ensures that the population does not deteriorate. After the selection is completed, the needSelection flag is reset.

Basic steps of the BSA algorithm (after initialization or selection): before generating new solutions, the current fitness values of all agents from "a" are saved in prevFitness for later use in Selection-II and the current coordinates of agents from "a [i].c" are copied to "a [i].cP" to allow for rollback if the new solutions turn out to be worse.

The internal SelectionI method is called. This step is responsible for selecting or preparing the "archive" oldP operation that will be used for mutation. The internal Mutation method is called. In this step, a "mutant" M population is generated based on the active and/or archived populations. The Crossover internal method is called. At this step, the M mutant population is mixed with the current 'a' active population, resulting in the formation of a "test" T population.

The coordinates of agents from the generated T trial population are copied into the active 'a' population. Now 'a' contains new "test" solutions whose fitness needs to be calculated. The needSelection flag is set to 'true'. This signals that greedy selection (Selection-II) should be performed on the next Moving call (after the fitness of the new solutions in 'a' has been calculated).

Thus, the Moving method encapsulates one complete iteration or "epoch" of the BSA algorithm, including initialization, selection, mutation, crossover, and preparation for result comparison.

//————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm
void C_AO_BSA_Backtracking::Moving ()
{
  // Initial population setup
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      }
    }

    revision      = true;
    needSelection = false;
    return;
  }

  // If you want to perform greedy selection after calculating fitness
  if (needSelection)
  {
    // Selection-II: Greedy selection
    for (int i = 0; i < popSize; i++)
    {
      // If the current solution (from T) is worse than the previous one, return the previous one
      if (a [i].f < prevFitness [i])
      {
        ArrayCopy (a [i].c, a [i].cP, 0, 0, WHOLE_ARRAY);
        a [i].f = prevFitness [i];
      }
    }

    needSelection = false;
  }

  //--- BSA basic steps:

  // Save current fitness before generating a new population 
  for (int i = 0; i < popSize; i++)
  {
    prevFitness [i] = a [i].f;
    ArrayCopy (a [i].cP, a [i].c, 0, 0, WHOLE_ARRAY);
  }

  // 1. Selection-I
  SelectionI ();

  // 2. Mutation
  Mutation ();

  // 3. Crossover
  Crossover ();

  // 4. Copy the trial population T into the 'a' main population to calculate fitness
  for (int i = 0; i < popSize; i++)
  {
    ArrayCopy (a [i].c, T [i].c, 0, 0, WHOLE_ARRAY);
  }

  // Set the flag to execute Selection-II after calculating fitness
  needSelection = true;
}
//————————————————————————————————————————————————————————————————————
The SelectionI method implements the first stage of selection in the BSA algorithm and is responsible for choosing the historical population that is subsequently used for mutation.

Probabilistic update of the historical population: with the probability of 50% (or, more simply, with equal probability) the oldP historical population is updated by the 'a' current population. If a random number is generated that is less than a given threshold (0.5), then the contents ('c', coordinates) of each agent from the current 'a' population are copied to the corresponding agent in the oldP historical population, and the 'f' fitness of the agent is also copied.

Shuffling historical population: Regardless of whether the historical population was updated in the previous step or not, the ShufflePopulation(oldP) method is called to shuffle the agents in the historical population. This is done to introduce randomness into the mutation. Shuffling ensures that agents in the historical population are selected in a random order, rather than in the order in which they were originally located.

Thus, SelectionI either updates the archive population with current candidate solutions or maintains its previous state, and then, in either case, shuffles the agents in the archive population, which allows for diversification of the search during subsequent mutation.

//————————————————————————————————————————————————————————————————————
//--- Selection-I: select a historical population
void C_AO_BSA_Backtracking::SelectionI ()
{
  // Update the historical population with a 50% probability 
  if (u.RNDprobab () < 0.5) // equivalent to if (a < b) where a,b ~ U(0,1)
  {
    // Copy the current population to the historical one
    for (int i = 0; i < popSize; i++)
    {
      ArrayCopy (oldP [i].c, a [i].c, 0, 0, WHOLE_ARRAY);
      oldP [i].f = a [i].f;
    }
  }

  // Shuffle the historical population
  ShufflePopulation (oldP);
}
//————————————————————————————————————————————————————————————————————
The ShufflePopulation method is designed to randomly shuffle elements in a given population of agents (represented by the S_AO_Agent structure). It takes as input an array of pop[] agents and performs in-place shuffling, i.e. changes the order of elements directly in the passed array.

Mixing algorithm: The method uses the Fisher-Yates shuffle algorithm to randomly mix elements in a population, the algorithm ensures that each permutation (order) of the elements has equal probability.

The "for" loop starts at the last element of the array (popSize - 1) and goes in reverse order to the second element (index 1). The first element (index 0) does not need to be shuffled, since by that time it will already have been "moved" during the algorithm. For each element with index i, a random index j is selected in the range from 0 to i inclusive. This is done using the u.RNDminusOne(i+1) function, which returns a random integer in the specified range (including"0 and excluding i+1).

Elements with i and j indices are swapped. For this purpose, the 'temp' temporary variable of the S_AO_Agent type is used. Since S_AO_Agent contains the 'c' array (coordinates), the array is copied. The f fitness value is also copied. The coordinate and fitness values of the pop[i] element are stored in the temp temporary variable, and the coordinate and fitness values of the pop[j] element are copied to pop[i]. The saved values from 'temp' (the original value of pop[i]) are copied to pop[j]. As a result, after the loop is completed, the order of elements in the pop[] array will be random.

//————————————————————————————————————————————————————————————————————
//--- Shuffle population
void C_AO_BSA_Backtracking::ShufflePopulation (S_AO_Agent &pop [])
{
  for (int i = popSize - 1; i > 0; i--)
  {
    int j = u.RNDminusOne (i + 1);

    // Swap i and j elements
    S_AO_Agent temp;
    temp.Init (coords);

    ArrayCopy (temp.c, pop [i].c, 0, 0, WHOLE_ARRAY);
    temp.f = pop [i].f;

    ArrayCopy (pop [i].c, pop [j].c, 0, 0, WHOLE_ARRAY);
    pop [i].f = pop [j].f;

    ArrayCopy (pop [j].c, temp.c, 0, 0, WHOLE_ARRAY);
    pop [j].f = temp.f;
  }
}
//————————————————————————————————————————————————————————————————————
The Mutation method is responsible for generating the mutant population, which is a key step in the BSA algorithm. It is based on the difference between the current population and the historical population.

First, a random amplitude factor of F is generated, which determines the magnitude of the impact of the historical population on the current one. Then, for each agent in the population and for each coordinate within that agent, a new value for the coordinate in the mutant population is calculated. This is done by adding to the current coordinate the product of the F amplitude factor and the difference between the corresponding coordinate in the historical population and the current coordinate. Thus, a mutant population is created by shifting the current population in the direction determined by the historical population, with an amplitude depending on the F factor.

//————————————————————————————————————————————————————————————————————
//--- Mutation: generation of a mutant population
void C_AO_BSA_Backtracking::Mutation ()
{
  // Generate the amplitude factor
  F = u.GaussDistribution (0.0, -3.0, 3.0, 2);

  // Apply mutation: M = P + F * (oldP - P)
  for (int i = 0; i < popSize; i++)
  {
    for (int j = 0; j < coords; j++)
    {
      M [i].c [j] = a [i].c [j] + F * (oldP [i].c [j] - a [i].c [j]);
    }
  }
}
//————————————————————————————————————————————————————————————————————
The Crossover method is involved in forming a trial population based on the current mutant population using the selected crossover strategy. This process aims to combine the features of inherited solutions to find more optimal options.

Initially, a trial population is copied from the mutant one to provide a basis for further modifications. Then a choice of crossing-over strategy occurs: with a 40% probability the first strategy is used, otherwise the second one.

If the first option is chosen, the mixrate strategy is implemented. In this case, for each agent, the number of elements (within the coordinates) that will be taken from the current agent is determined, taking into account the specified mixrate coefficient. These elements are selected randomly, without repetition. After this, the corresponding coordinates from the current solution are copied from the selected indices to the agent of the trial population.

If the second strategy is chosen, then for each agent one random element (coordinate) is selected, and in the trial population in the corresponding position this element is replaced by its value from the current solution.

After crossing over is completed, bounds checking is performed for each agent in the trial population to ensure that all decisions are valid and meet the range requirements.

//————————————————————————————————————————————————————————————————————
//--- Crossover: trial population generation
void C_AO_BSA_Backtracking::Crossover ()
{
  // Initialize the trial population as a copy of the mutant one
  for (int i = 0; i < popSize; i++)
  {
    ArrayCopy (T [i].c, M [i].c, 0, 0, WHOLE_ARRAY);
  }

  // Select a crossover strategy
  if (u.RNDprobab () < 0.4)
  {
    //--- STRATEGY 1: Using mixrate
    for (int i = 0; i < popSize; i++)
    {
      // Reset the map
      ArrayInitialize (map [i].val, 0);

      // Define the number of elements for the crossover
      int numElements = (int)MathCeil (mixrate * u.RNDprobab () * coords);

      // Generate unique indices for the crossover
      for (int n = 0; n < numElements; n++)
      {
        int idx;
        do
        {
          idx = u.RNDminusOne (coords);
        }
        while (map [i].val [idx] == 1); // until we find an unused index

        map [i].val [idx] = 1;
      }

      // Apply crossover
      for (int j = 0; j < coords; j++)
      {
        if (map [i].val [j] == 1)
        {
          T [i].c [j] = a [i].c [j];
        }
      }
    }
  }
  else
  {
    //--- STRATEGY 2: Mutation of only one element
    for (int i = 0; i < popSize; i++)
    {
      // Select one random element
      int randomIndex = u.RNDminusOne (coords);
      T [i].c [randomIndex] = a [i].c [randomIndex];
    }
  }

  // Boundary control for all agents in the trial population
  for (int i = 0; i < popSize; i++)
  {
    BoundaryControl (T [i]);
  }
}
//————————————————————————————————————————————————————————————————————
The BoundaryControl method is designed to check and adjust the agent's parameter values to ensure they remain within acceptable limits, as well as to convert the decisions to the desired discrete format.

For each element of the agent's coordinates, a check is performed: if the value goes beyond the established minimum or maximum limits, then this output is processed in accordance with the selected strategy. If the probability is randomly selected below 50%, a random regeneration of a value within the acceptable range occurs. Otherwise, the value is set to the nearest boundary - either the minimum or the maximum.

After this, each coordinate value is discretized — that is, converted into the closest acceptable value corresponding to the specified range and discretization step. This ensures that the solution meets the data type and range requirements.

//————————————————————————————————————————————————————————————————————
//--- Boundary control
void C_AO_BSA_Backtracking::BoundaryControl (S_AO_Agent &agent)
{
  for (int j = 0; j < coords; j++)
  {
    if (agent.c [j] < rangeMin [j] || agent.c [j] > rangeMax [j])
    {
      // Select a boundary handling strategy
      if (u.RNDprobab () < 0.5)
      {
        // Random regeneration
        agent.c [j] = u.RNDfromCI (rangeMin [j], rangeMax [j]);
      }
      else
      {
        // Set to the boundary
        if (agent.c [j] < rangeMin [j]) agent.c [j] = rangeMin [j];
        else agent.c [j] = rangeMax [j];
      }
    }

    // Discretization
    agent.c [j] = u.SeInDiSp (agent.c [j], rangeMin [j], rangeMax [j], rangeStep [j]);
  }
}
//————————————————————————————————————————————————————————————————————
The Revision method is responsible for selecting and updating the best solution found in the current population. Going through all the solutions, it searches for the one with the maximum value of the evaluation function (fitness). If such a solution is found, it is stored as the current best result, and its coordinates are copied into a separate array designed to store the best solution at the moment. The method thus provides continuous tracking and updating of the optimal solution found during the algorithm iterations.

//————————————————————————————————————————————————————————————————————
//--- Selection-II and updating the best solution
void C_AO_BSA_Backtracking::Revision ()
{
  int bestIND = -1;

  for (int i = 0; i < popSize; i++)
  {
    // Update the global best solution
    if (a [i].f > fB)
    {
      fB = a [i].f;
      bestIND = i;
    }
  }

  // Copy the coordinates of the best solution
  if (bestIND != -1)
  {
    ArrayCopy (cB, a [bestIND].c, 0, 0, WHOLE_ARRAY);
  }
}
//————————————————————————————————————————————————————————————————————
The GaussDistribution method generates a random number with a Gaussian (normal) distribution centered around a given input value and bounded by a certain range, using the Box-Muller method for producing normally distributed random variables.

First, it generates two uniformly distributed random variables. Then, based on these values, a standard normally distributed random variable is calculated, which can be used to obtain a Gaussian distribution.

The method then checks whether the generated value is within the specified deviation range defined by the sigma parameter. If the generated value is outside these limits, the method is called recursively again to generate a new value until it falls within the allowed range.

Finally, the generated normally distributed variance is scaled to fit within the given output range (from outMin to outMax) relative to the In input value. This allows us to shift and scale the distribution to match your desired parameters. The result is a Gaussian distributed number centered at In, but subject to constraints on the minimum and maximum output value.

//——————————————————————————————————————————————————————————————————————————————
double C_AO_Utilities :: GaussDistribution (const double In, const double outMin, const double outMax, const double sigma)
{
  double logN = 0.0;
  double u1   = 2.0 * MathRand () / 32767.0 - 1.0;//RNDfromCI (0.0, 1.0);
  double u2   = 2.0 * MathRand () / 32767.0 - 1.0;//RNDfromCI (0.0, 1.0);

  logN = u1 <= 0.0 ? 0.000000000000001 : u1;

  double z0 = sqrt (-2 * log (logN)) * cos (2 * M_PI * u2);

  double sigmaN = sigma > 8.583864105157389 ? 8.583864105157389 : sigma;
  
  // If z0 is outside the range [-sigmaN, sigmaN], generate anew
  if (z0 >= sigmaN || z0 <= -sigmaN) 
  {
    return GaussDistribution(In, outMin, outMax, sigma); // Recursive call
  }

  if (z0 >= 0.0) z0 =  Scale (z0,        0.0, sigmaN, 0.0, outMax - In, false);
  else           z0 = -Scale (fabs (z0), 0.0, sigmaN, 0.0, In - outMin, false);
  
  return In + z0;
}
//——————————————————————————————————————————————————————————————————————————————


Test results
BSA algorithm shows very good results.

BSA|Backtracking Search Algorithm|10.0|1.0|
=============================
5 Hilly's; Func runs: 10000; result: 0.9730917210619289
25 Hilly's; Func runs: 10000; result: 0.5453406317593932
500 Hilly's; Func runs: 10000; result: 0.2909827609772065
=============================
5 Forest's; Func runs: 10000; result: 0.9999986842258451
25 Forest's; Func runs: 10000; result: 0.5854340780208712
500 Forest's; Func runs: 10000; result: 0.21747482800959225
=============================
5 Megacity's; Func runs: 10000; result: 0.8476923076923077
25 Megacity's; Func runs: 10000; result: 0.3695384615384615
500 Megacity's; Func runs: 10000; result: 0.12978461538461658
=============================
All score: 4.95934 (55.10%)
In the visualization of the BSA algorithm, one can notice the spread of values in the results for functions of small and medium dimensions; however, with an increase in the population size parameters, the spread decreases, but for better convergence, it is necessary to increase the number of iterations.


READ 6
Dolphin Echolocation Algorithm (DEA)
MetaTrader 5 — Trading | 2 June 2026 at 18:20
1 669
0
Andrey Dik
Andrey Dik
Contents
Introduction
Implementation of the algorithm
Test results


Introduction
With each new article, we get closer to our goal: finding a suitable algorithm for solving optimization problems with our trading robots. We will explore another nature-inspired algorithm based on the echolocation capabilities of some marine animals.

Imagine a dolphin hunting in the dark depths of the ocean. Visibility is practically zero, but this does not prevent it from successfully finding prey. The secret lies in an amazing ability: the dolphin emits a series of clicks and, using the reflected echo, creates an accurate picture of the surrounding space. An interesting fact is that the frequency of these clicks varies: rare clicks during a general search and frequent ones when the target is close. It is this unusual strategy that forms the basis of the DEA (Dolphin Echolocation Algorithm).

In the world of optimization, we often face a similar challenge: finding the best solution in a vast space of possibilities without having the full picture. Like a dolphin in the ocean, we start with a broad search and gradually focus on the most promising areas.



Implementation of the algorithm
To better understand how the algorithm works, let's imagine the following situation. You and your friends are looking for gold on a large beach, armed with metal detectors. At the beginning of the search, it makes sense to spread out across the entire area — this way you have a better chance of stumbling upon something interesting. But as soon as one of you hears a strong signal, they inform the others, and gradually the whole team begins to concentrate in promising places. By the end of the search, everyone is digging near the strongest signal. This is the essence of the dolphin echolocation algorithm.

In the algorithm, the role of dolphins is played by search agents - points in the solution space. Each such "dolphin" represents a potential solution to the problem. For example, if we are looking for the minimum of a simple function y = x², then one dolphin may be at the point x = -3 (where y = 9), another at the point x = 1 (where y = 1), and the third will randomly end up at the point x = 0 (where y = 0) - this will be our champion.

But how do dolphins exchange information? This is where the concept of effective radius, denoted as "Re", comes into play. Think about how far the light from a flashlight spreads. At Re = 1 we have a narrow beam that illuminates only the immediate area. At Re = 3, the light spreads wider, covering more space. At Re = 5 or above, the search behaves more like a floodlight. In the context of the algorithm, this means that information about a good solution spreads to neighboring areas, and the strength of this influence decreases with distance.

All this information is accumulated in the form of a "prospectivity map", which the algorithm calls the accumulated fitness (AF). Imagine a heat map of a city, where "hot" zones indicate areas of high activity. In our case, "hot" zones are areas where dolphins have found good solutions (prey). The more successful finds in a particular area, the "hotter" it becomes, attracting other dolphins.

However, the algorithm is smarter than simply clustering in one place. It uses a parameter called Predetermined Probability (PP) which controls the balance between exploring new areas and exploiting good spots already found. At the beginning, when PP is low, around 0.1, the algorithm experiments, and then, when PP is around 0.5, it relies more on proven approaches, and, as PP approaches 1, it uses only the most effective methods.

Let's look at a specific example of how the algorithm works. Let's say we are looking for the highest point in a hilly area. At the beginning of the search, our dolphins are randomly scattered throughout the area - some in the valley, some on the slopes, and some are lucky enough to be on top of a hill. After evaluating the height of each position, the algorithm determines that the dolphin at the top has found the best spot. Now something interesting happens: the area around this lucky dolphin becomes more attractive to the others, but – and this is the key point – the very point where the leader is located is temporarily "zeroed out". This forces other dolphins to explore nearby areas rather than congregate in one spot. This approach helps to avoid premature convergence and find other potentially good solutions.

As the algorithm works, the picture changes. At iteration 50, we see that the dolphins begin to cluster in hilly areas, but still retain some diversity. By the 100th iteration, most of them are concentrated at the highest points of the terrain, and at the end, all dolphins are concentrated at the global maximum. The efficiency of the algorithm depends on the correct configuration of the parameters.

The choice of the effective Re radius is important to balance between local and global search. At Re = 1, we get a very precise but narrowly focused search – as if we were looking for a needle in a haystack with a magnifying glass. Increasing Re to 3-4 gives a balanced approach, similar to searching with a flashlight. And when Re is greater than 5, the algorithm works like a spotlight, covering larger areas, but losing accuracy.

The Power parameter determines how abruptly the algorithm switches from exploration to exploitation. At Power = 1 this transition is smooth and gradual. A Power value of 2 (recommended) produces a more pronounced transition, while Power values of 3 and higher produce a sharper transition, which can be useful for certain types of tasks.

The initial probability of PP1 sets the starting balance between exploration and exploitation. A low value (0.05) means aggressive exploration of the space, the standard value of 0.1 provides a good balance, and a high value (0.5) leads to a quick focus on the solutions found.

The main advantage of DEA is its adaptability — the algorithm automatically adjusts the balance between exploring new areas and delving into promising zones. Yet it remains relatively simple, requiring only four settings to be configured. The mechanism of information propagation through accumulated fitness allows for the efficient use of knowledge about the search space, and zeroing the AF for the best positions helps avoid getting stuck in local optima.

Of course, the algorithm has some limitations. It requires additional memory to store the accumulated fitness of all alternatives, which can be a problem for very large search spaces. And perhaps one more thing: the efficiency of the algorithm depends on the correct choice of the Re effective radius. Below is a schematic illustration of the algorithm operation.

dea_algorithm

Figure 1. DEA algorithm in operation

 The main stages of the DEA algorithm are shown in the figure:

Initial exploration - dolphins (search agents) in random positions with Re echolocation radius.
Accumulated fitness (AF) distribution - How "AF" accumulates around dolphin positions.
The convergence process consists of three sub-stages that show the transition from exploration to exploitation.
The illustration helps us understand how dolphins use echolocation to find the optimum, how information is disseminated through accumulated fitness, how the parameter "Re" affects the search width, and how "PP" controls the balance between exploration and exploitation. Key concepts — formulas for PP (predetermined probability) and AF. Algorithm flow chart  — basic steps with a cycle. Influence of the Re parameter — visualization of narrow, medium and wide radius of influence.

The color scheme in the figure: blue - normal dolphins, red - the best solution, blue gradients - accumulated fitness levels, gray - search space. 

Now that we have some idea of the algorithm, we can move on to writing detailed pseudocode. 

Inputs:
Number of dolphins (search agents)
Effective echolocation radius
Degree of convergence curve
Initial predetermined probability
Search space boundaries and sampling step for each dimension
Initialization:
Creating a space of alternatives:

For each dimension of the search space, generate a set of possible alternatives
If a sampling step is specified, create alternatives with this step from minimum to maximum
If step is not specified, generate 500 uniformly distributed alternatives
Check the effective radius: it should not exceed a quarter of the number of alternatives
Initial placement of dolphins:

Randomly place all dolphins in the search space
For each dolphin, calculate the quality of its position (fitness)
Initialize the accumulated fitness to zero for all alternatives
Basic optimization loop:
Repeat for a specified number of iterations:

Phase 1. Calculating the predetermined probability

Calculate the probability of maintaining the best position at the current iteration. At the beginning of the process, this probability is equal to the initial value, then gradually increases according to a power function up to unity by the end of optimization.

Phase 2. Calculation of the dynamic adaptation coefficient

Calculate the coefficient that determines the balance between local and global search. The coefficient is equal to the ratio of the difference between the best and worst solutions to the sum of the deviations of all solutions from the best. When the population is diverse, the coefficient is high; when it converges, it is low.

Phase 3. Accumulation of information on fitness

For each dolphin:

Normalize its fitness relative to the current range in the population
For each coordinate, find the closest alternative to the dolphin's position
Spread quality information within the echolocation radius of this alternative
The strength of influence decreases linearly with distance from the center
Apply reflection at the boundaries of space to preserve information
Add a small value to all accumulated fitnesses to avoid zero probabilities
Step 4. Resetting information for a better position

Find the dolphin with the best solution and reset the accumulated fitness for the alternatives corresponding to its coordinates. This prevents the search from becoming too concentrated in one point.

Step 5. Selecting new positions

For each dolphin and each of its coordinates:

If this is the best dolphin and there is a chance to save the position, leave the coordinate unchanged
Otherwise, if there is accumulated information about fitness, select a new alternative in proportion to this information using the roulette method
If the accumulated information is missing or insufficient:
with a probability equal to the dynamic coefficient, perform a local search within the echolocation radius
otherwise, perform a global search by selecting a random alternative
Apply search space constraints and discretization
Stage 6. Evaluation of new positions

Calculate the quality of the solution for each dolphin in its new position.

Stage 7. Global information update

Update the records of the best and worst solutions in the population. If a solution is found that is better than the current global optimum, we should keep it.

Completion:
After completing all iterations, return the best solution found and its quality.

Let us now proceed to the DEA implementation.

Let's write the S_Alternative structure. It stores information about an alternative in the decision-making process for optimization and contains "value" (the value characterizing this alternative, an assessment of the effectiveness and AF) - the accumulated fitness for this alternative, this is necessary for assessing the quality of the alternative when it is necessary to apply an iterative process.

//————————————————————————————————————————————————————————————————————
struct S_Alternative
{
    double value;     // alternative value
    double AF;        // accumulated fitness for this alternative
};
//————————————————————————————————————————————————————————————————————
The S_Coordinate structure is intended to represent a set of alternatives associated with a specific coordinate, "alt" is an array of alternatives, each of which corresponds to a coordinate, "count" is a variable indicating the current number of alternatives actually stored in the "alt" array.

//————————————————————————————————————————————————————————————————————
struct S_Coordinate
{
    S_Alternative alt [];  // array of alternatives for a given coordinate
    int           count;   // number of alternatives
};
//————————————————————————————————————————————————————————————————————
Next, we move on to the description of the class that directly implements the optimization algorithm (DEA). The C_AO_DEA class is an inheritor of the C_AO base class. When creating a class object, the main parameters of the algorithm are initialized:

popSize — initializes the population size (number of "dolphins" or locations).
Re — initial value of the effective search radius.
Power — convergence curve degree.
PP1 — initializes the convergence factor for the first iteration.
Then the "params" array is initialized, which is used to store the user parameters of the algorithm. The initial values popSize, Re, Power, PP1 with the corresponding names are copied into it.
The SetParams method is designed to update the internal parameters of the algorithm based on the values written in the params array. After extracting popSize, Re, Power, PP1 values, a validation check is performed.

Methods:

Init is designed to initialize the algorithm, accepting minimum, maximum values and steps for each parameter, as well as the total number of epochs (iterations).
Moving is responsible for moving the "dolphins" in the search space; this is part of the main optimization loop.
Revision adjusts population positions and parameters after movement.
The following fields are used internally to run the algorithm and are not accessible from outside the class.

PP is a floating-point number representing a predefined probability for the current iteration, used for stochastic decisions, currentEpoch is a value tracking the current epoch (iteration) of the algorithm.
totalEpochs is a value that stores the total scheduled number of epochs.
coeffA is a dynamic coefficient used to select positions.
coordData is an array of structures, where each S_Coordinate contains an array of alternatives and their number for a particular coordinate. 
Methods:
CalculatePP is used to calculate the PP value (predetermined probability) at the current iteration.
CalculateAccumulativeFitness calculates the accumulated fitness for each alternative.
ResetAFforBestLocation resets the accumulated fitness values for best locations.
SelectNextLocations selects the next locations for the dolphins based on the current state.
FindNearestAlternative finds the nearest alternative based on the given coordinates and value.
CalculateCoefficientA calculates the coeffA dynamic coefficient.
In general, the C_AO_DEA class is ready to be used for finding solutions in multidimensional space. It has public methods for initializing and performing the main optimization steps, as well as private methods and data that implement the internal logic of the algorithm.

//————————————————————————————————————————————————————————————————————
class C_AO_DEA : public C_AO
{
  public: //----------------------------------------------------------
  ~C_AO_DEA () { }
  C_AO_DEA ()
  {
    ao_name = "DEA";
    ao_desc = "Dolphin Echolocation Algorithm";
    ao_link = "https://www.mql5.com/en/articles/18495";

    popSize = 100;    // NL - number of locations (dolphins) 
    Re      = 2;      // effective search radius
    Power   = 2.0;    // convergence curve power
    PP1     = 1.0;    // first iteration convergence factor

    ArrayResize (params, 4);

    params [0].name = "popSize"; params [0].val = popSize;
    params [1].name = "Re";      params [1].val = Re;
    params [2].name = "Power";   params [2].val = Power;
    params [3].name = "PP1";     params [3].val = PP1;
  }

  void SetParams ()
  {
    popSize = (int)params [0].val;
    Re      = (int)params [1].val;
    Power   = params      [2].val;
    PP1     = params      [3].val;

    // Check the parameters validity
    if (Re < 0) Re = 0;
    if (PP1 < 0.0) PP1 = 0.0;
    if (PP1 > 1.0) PP1 = 1.0;
    if (Power < 0.1) Power = 0.1;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //------------------------------------------------------------------
  int    Re;           // effective search radius
  double Power;        // convergence curve power
  double PP1;          // first iteration convergence factor

  private: //---------------------------------------------------------
  double PP;           // predefined probability for the current iteration
  int    currentEpoch; // current epoch
  int    totalEpochs;  // total number of epochs
  double coeffA;       // dynamic coefficient used to select positions

  S_Coordinate coordData []; // coordinate data (alternatives and AF)

  void CalculatePP ();
  void CalculateAccumulativeFitness ();
  void ResetAFforBestLocation ();
  void SelectNextLocations    ();
  int  FindNearestAlternative (int coord, double value);
  void CalculateCoefficientA  ();
};
//————————————————————————————————————————————————————————————————————
The Init method of the C_AO_DEA class initializes the algorithm (DEA). It takes the minimum and maximum values for each variable, the steps to change each variable, and the total number of epochs to optimize as inputs. The method first calls the base StandardInit method to perform standard initialization of the algorithm common parameters and, if this initialization fails, returns 'false'. Then variables that track the current and total epochs of the algorithm execution are initialized.

After this, the coordData data structure is created to store information about each coordinate (variable) of the search space. For each coordinate, the number of possible alternative values is determined. If a step change is specified for a coordinate, the number of alternatives is calculated based on the boundaries and the step. If the step is not specified (equal to zero), the coordinate is assumed to be continuous and a fixed number of alternatives is set for it.

Then the Re parameter (search radius) is checked and, if necessary, adjusted so that it does not exceed a quarter of the number of alternatives for each coordinate. After this, memory is allocated to store alternative values for each coordinate.

Finally, the loop fills an array of alternative values for each coordinate, distributing them evenly between the minimum and maximum values. If a step is specified, alternative values are calculated using that step. If the step is not specified, discretization of the space between the specified boundaries is used. Also, for each alternative, the associated AF (accumulated fitness) value is initialized to zero. If all initialization steps are successfully completed, the method returns 'true'.

//————————————————————————————————————————————————————————————————————
//--- Initialization
bool C_AO_DEA::Init (const double &rangeMinP  [],
                     const double &rangeMaxP  [],
                     const double &rangeStepP [],
                     const int epochsP = 0)
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //------------------------------------------------------------------
  currentEpoch = 0;
  totalEpochs  = epochsP;

  // Initialize the data structure for coordinates
  ArrayResize (coordData, coords);

  // Create alternatives for each coordinate
  for (int c = 0; c < coords; c++)
  {
    if (rangeStep [c] != 0)
    {
      coordData [c].count = (int)((rangeMax [c] - rangeMin [c]) / rangeStep [c]) + 1;
    }
    else
    {
      coordData [c].count = 500;
    }

    // Check that Re is not too large for the number of alternatives
    if (Re > coordData [c].count / 4) Re = coordData [c].count / 4;

    ArrayResize (coordData [c].alt, coordData [c].count);

    // Fill in the alternatives
    for (int i = 0; i < coordData [c].count; i++)
    {
      if (rangeStep [c] != 0)
      {
        coordData [c].alt [i].value = rangeMin [c] + i * rangeStep [c];
      }
      else
      {
        coordData [c].alt [i].value = rangeMin [c] + (rangeMax [c] - rangeMin [c]) * i / (coordData [c].count - 1);
      }
      coordData [c].alt [i].AF = 0.0;
    }
  }

  return true;
}
//————————————————————————————————————————————————————————————————————
The Moving method implements one main step of the algorithm (DEA) corresponding to a certain iteration of the optimization process. If this is the first iteration detected using the "revision" flag, an initial random assignment of population coordinates occurs. For each section (population element) and for each variable in the range, random values are assigned. These values are adjusted taking into account the boundaries and the step of change to ensure acceptable solutions. After this, the flag is set, initialization is done, and the method exits this loop.

Once the initialization is done, the current epoch (iteration) counter is incremented by one. The key steps of the algorithm are then performed: performance metrics are determined for each solution in the current population, which indicate how well they fit the problem. The "a" coefficient is calculated. Based on the quality indicators, a generalized measure of the fitness of each solution is determined and the "AF" index is reset for it, which allows focusing the search in the area of the best solutions. Based on current information and calculated coefficients, the next locations (positions) are selected for updating and moving candidate solutions in the search space.

In general, the Moving method implements one iteration cycle of the DEA algorithm, sequentially performing key steps of updating and improving solutions in the process of searching for the optimum.

//————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm (according to Algorithm 1)
void C_AO_DEA::Moving ()
{
  // Initial setup
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        a [p].cB [c] = a [p].c [c];
      }
    }

    revision = true;
    return;
  }

  // Increase the epoch counter
  currentEpoch++;

  // Steps of the DEA algorithm according to Algorithm 1:

  // 1. Calculate PP for the current iteration
  CalculatePP ();

  // 2. We calculate the 'a' dynamic coefficient
  CalculateCoefficientA ();

  // 3. Calculate accumulated fitness
  CalculateAccumulativeFitness ();

  // 4. Find the best location and reset AF for it
  ResetAFforBestLocation ();

  // 5. Select the next positions
  SelectNextLocations ();
}
//————————————————————————————————————————————————————————————————————
The CalculatePP method computes the predetermined probability (PP), which is used in the decision-making process within the algorithm. If the total number of epochs (iterations) is equal to one or less, then the probability is set equal to a predetermined initial value (PP1). In this case, no further calculation is required and the method terminates.

If the number of epochs is greater than one, then the PP value is calculated based on a given formula that takes into account the current epoch and includes the following logic: the power, which is a function of the current epoch raised to the power (Power), is calculated, and one is subtracted from this, similarly the power is calculated for the total number of epochs.

The PP value is updated using a formula that gradually approaches 1, taking into account the current progress across epochs. Specifically, a portion proportional to the progress adjusted by the degree and the given Power and totalEpochs parameters is added to the initial PP1 value. If the overall exponent is non-zero, division occurs to obtain the current probability; otherwise, the value remains equal to the initial PP1.

Ultimately, the method dynamically changes the probability value during the execution of the algorithm, ensuring a balance between the initial and more "progressive" values as the epochs progress.

//————————————————————————————————————————————————————————————————————
//--- Calculate the predetermined probability (according to the formula 5)
void C_AO_DEA::CalculatePP ()
{
  if (totalEpochs <= 1)
  {
    PP = PP1;
    return;
  }

  // PP = PP1 + (1 - PP1) * ((Loop^Power - 1) / (LoopsNumber^Power - 1))
  double iterPower  = MathPow ((double)currentEpoch, Power) - 1.0;
  double totalPower = MathPow ((double)totalEpochs,  Power) - 1.0;

  if (totalPower != 0)
  {
    PP = PP1 + (1.0 - PP1) * iterPower / totalPower;
  }
  else
  {
    PP = PP1;
  }
}
//————————————————————————————————————————————————————————————————————
CalculateCoefficientA computes the 'a' dynamic coefficient used to regulate the search process. The method goes through the entire current population of solutions and for each solution calculates the difference between the maximum possible fB fitness and the current fitness of the particular solution (a[i].f). These differences accumulate into a sum.

After accumulating the sum of values, the 'a' coefficient is obtained as the ratio of the difference between fB and the minimum fW fitness to the resulting sum. This allows us to define a dynamic coefficient that adapts depending on the current state of the population and helps balance the rate of convergence or exploration of the search space.
This method thus creates a scaling parameter that takes into account the fitness distribution of solutions and influences their updating and moving strategy in subsequent iterations of the algorithm.

//————————————————————————————————————————————————————————————————————
//--- Calculate the dynamic 'a' coefficient
void C_AO_DEA::CalculateCoefficientA ()
{
  double sumFitness = 0.0;

  for (int i = 0; i < popSize; i++)
  {
    sumFitness += fB - a [i].f;
  }

  coeffA = (fB - fW) / sumFitness;
}
//————————————————————————————————————————————————————————————————————
The FindNearestAlternative method is designed to find the nearest alternative to a given value within a specific coordinate and takes two arguments: the "coord" coordinate index and the "value" we need the nearest alternative for. Initial values for "nearest" (the index of the nearest alternative) and "minDist" (the minimum distance) are initialized and set.

The loop goes through all alternatives associated with the given coord coordinate. For each alternative, the distance between the given "value" and the alternative value (coordData[coord].alt[i].value) is calculated, and if the calculated distance is less than the current minimum distance (minDist), then "minDist" is updated with the new, smaller distance value, and "nearest" is updated with the index of the current alternative.

After the loop is completed, the index of the alternative that was closest to the given "value" within the given coordinate is returned. Thus, the method determines which of the available alternatives for the specified coordinate is closest to the given real value.

//————————————————————————————————————————————————————————————————————
//--- Search for the closest alternative
int C_AO_DEA::FindNearestAlternative (int coord, double value)
{
  int nearest = 0;
  double minDist = DBL_MAX;

  for (int i = 0; i < coordData [coord].count; i++)
  {
    double dist = MathAbs (value - coordData [coord].alt [i].value);
    if (dist < minDist)
    {
      minDist = dist;
      nearest = i;
    }
  }

  return nearest;
}
//————————————————————————————————————————————————————————————————————
The CalculateAccumulativeFitness method is designed to calculate the accumulated fitness (AF) for alternatives according to Algorithm 2. Before starting the calculations, all accumulated fitness values for each alternative in each coordinate are cleared and set to zero. The range of fitnesses in the current population of solutions is found, calculated as the difference between the maximum possible fitness (fB) and the minimum (fW).

Then, for each agent (dolphin), its fitness is normalized relative to the range, and for each search coordinate, the closest alternative is determined, and in the radius "Re" around this alternative, the accumulated fitness for the alternative is updated, taking into account the reflective nature of the boundaries. This is done by weighted addition of contributions, where the weights are formed based on the distance from the nearest alternative (taking into account reflective boundaries), and include a coefficient associated with the current step in radius "Re".

As a result of the method operation, accumulated fitness values are formed for each alternative in each coordinate, taking into account the contribution of neighboring alternatives.

//————————————————————————————————————————————————————————————————————
//--- Calculate the accumulated fitness (according to Algorithm 2)
void C_AO_DEA::CalculateAccumulativeFitness ()
{
  // Clear the accumulated fitness for all alternatives
  for (int c = 0; c < coords; c++)
  {
    for (int i = 0; i < coordData [c].count; i++)
    {
      coordData [c].alt [i].AF = 0.0;
    }
  }

  double rangeFF = fB - fW;
  if (rangeFF == 0.0) rangeFF = DBL_EPSILON;

  // For each agent (dolphin)
  for (int i = 0; i < popSize; i++)
  {
    // Normalize 'fitness' for this agent
    double normalizedFitness = (a [i].f - fW) / rangeFF;

    for (int c = 0; c < coords; c++)
    {
      // Find the closest alternative for the current coordinate
      int nearestAlt = FindNearestAlternative (c, a [i].c [c]);

      // Update the accumulated fitness in Re radius
      // According to Algorithm 2: AF(A+k)j = (1/Re) * (Re - |k|) * fitness(i) + AF(A+k)j
      for (int k = -Re; k <= Re; k++)
      {
        int altIndex = nearestAlt + k;

        // Boundary check taking into account reflection (reflective characteristic)
        if (altIndex < 0)
        {
          altIndex = -altIndex; // reflection from the lower border
        }
        else if (altIndex >= coordData [c].count)
        {
          altIndex = 2 * (coordData [c].count - 1) - altIndex; // reflection from the upper border
        }

        if (altIndex >= 0 && altIndex < coordData [c].count)
        {
          double weight = (1.0 / (double)(Re + 1)) * (Re - MathAbs (k) + 1);
          coordData [c].alt [altIndex].AF += weight * normalizedFitness;
        }
      }
    }
  }

  // Add a small epsilon value to all AFs to avoid zero probabilities
  double epsilon = 0.0001;
  for (int c = 0; c < coords; c++)
  {
    for (int i = 0; i < coordData [c].count; i++)
    {
      coordData [c].alt [i].AF += epsilon;
    }
  }
}
//————————————————————————————————————————————————————————————————————

The ResetAFforBestLocation method is designed to reset the accumulated fitness (AF) for alternatives corresponding to the location of the best solution (agent) in the population. First, the method determines the index of the best solution (agent) in the current population. It iterates through all solutions, finding the solution with the maximum fitness value. The index of the solution with the highest fitness is stored in the bestIndex variable.

For each 'c' coordinate of the best solution, the method determines the nearest alternative to the coordinate value of the best solution at that coordinate using the FindNearestAlternative function, and if the found alternative exists (the index is within the acceptable range), then the accumulated fitness value AF for that particular alternative is reset to zero.

Thus, the method resets AF only for those alternatives that most closely match the coordinates of the best solution. This is done to reduce the influence of these alternatives on subsequent stages of the search, potentially encouraging exploration of other areas of the search space.

As a result, the fitness of those coordinates that best correspond to the best solution is "zeroed/reset" to stimulate the search for other, possibly more optimal solutions.

//————————————————————————————————————————————————————————————————————
//--- Reset AF for the best location (according to Algorithm 3)
void C_AO_DEA::ResetAFforBestLocation ()
{
  // Find the index of the best solution
  int bestIndex = 0;
  double bestFitness = a [0].f;

  // Search for a solution with maximum fitness (since we always maximize normalized fitness)
  for (int i = 1; i < popSize; i++)
  {
    if (a [i].f > bestFitness)
    {
      bestFitness = a [i].f;
      bestIndex = i;
    }
  }

  // Reset AF for ALL alternatives corresponding to the coordinates of the best solution
  for (int c = 0; c < coords; c++)
  {
    // Find the closest alternative to the coordinate of the best solution
    int nearestAlt = FindNearestAlternative (c, a [bestIndex].c [c]);

    // Reset AF only for this alternative
    if (nearestAlt >= 0 && nearestAlt < coordData [c].count)
    {
      coordData [c].alt [nearestAlt].AF = 0.0;
    }
  }
}
//————————————————————————————————————————————————————————————————————
The SelectNextLocations method is designed to select the next position for each solution in the population (for each dolphin) based on probabilistic considerations and taking into account both the accumulated fitness (AF) and the strategy for maintaining the best position. The method first determines the best solution in the current population based on its fitness value. The best solution index is saved for further use.

For each solution and each of its coordinates, the following set of actions is performed: if the current solution is the best, and the random number is less than the PP probability, the current coordinate of this solution is kept unchanged. This preserves the previous best solution. However, if the current solution is not the best, or if the position is not preserved under PP, then all AF values for the alternatives at the current coordinate are summed up. If the total AF sum is greater than a small number (epsilon), indicating the presence of non-zero fitness values, a roulette is performed: a random number is chosen proportional to the AF sum, and the solution coordinate is chosen based on the cumulative AF sum of the alternatives, allowing solutions to move towards locations with greater fitness.

If the sum of AF is close to zero (all values of AF are very small), a local search is performed if the random number is less than the coeffA dynamic coefficient. In this case, a random offset (-Re...+Re) relative to the current value is selected, and the coordinate is updated with the nearest value. Boundaries are taken into account.

Global search (random selection) if random number is greater than coeffA. In this case, a completely random alternative for the coordinate is selected. At the end of the method, the obtained coordinate value is limited to the allowed range (rangeMin, rangeMax) and discretized with the given rangeStep step to ensure that the value is within the allowed range and corresponds to the allowed values.

As a result of this method, the coordinates of each solution are updated taking into account the probability weights based on the accumulated fitness, the PP (preserve the best) strategy, dynamic local and global search, which allows the algorithm to effectively explore the search space and find optimal solutions.

//————————————————————————————————————————————————————————————————————
//--- Select next positions based on probabilities
void C_AO_DEA::SelectNextLocations ()
{
  // First we find the index of the best solution
  int bestIndex = 0;
  double bestFitness = a [0].f;

  for (int i = 1; i < popSize; i++)
  {
    if (a [i].f > bestFitness)
    {
      bestFitness = a [i].f;
      bestIndex = i;
    }
  }

  for (int i = 0; i < popSize; i++)
  {
    for (int c = 0; c < coords; c++)
    {
      // For the best agent, apply PP
      if (i == bestIndex && u.RNDprobab () < PP)
      {
        // Save the current value of the coordinate of the best solution with PP probability
        continue;
      }

      // Select based on accumulated fitness
      double totalAF = 0.0;
      for (int alt = 0; alt < coordData [c].count; alt++)
      {
        totalAF += coordData [c].alt [alt].AF;
      }

      if (totalAF > DBL_EPSILON) // Check that there are non-zero AFs
      {
        // Choose an alternative based on roulette
        double rnd = u.RNDprobab () * totalAF;
        double cumSum = 0.0;

        for (int alt = 0; alt < coordData [c].count; alt++)
        {
          cumSum += coordData [c].alt [alt].AF;
          if (cumSum >= rnd)
          {
            a [i].c [c] = coordData [c].alt [alt].value;
            break;
          }
        }
      }
      else
      {
        // If all AFs are almost zero, use random selection
        // with the dynamic coeffA coefficient for the local search probability
        if (u.RNDprobab () < coeffA) // Use the dynamic coefficient
        {
          // Local search - stay close to the current position
          int currentAlt = FindNearestAlternative (c, a [i].c [c]);
          int shift = u.RNDminusOne (2 * Re + 1) - Re; // random offset within Re
          int newAlt = currentAlt + shift;

          // Check boundaries
          if (newAlt < 0) newAlt = 0;
          if (newAlt >= coordData [c].count) newAlt = coordData [c].count - 1;

          a [i].c [c] = coordData [c].alt [newAlt].value;
        }
        else
        {
          // Global search - completely random selection
          int randAlt = u.RNDminusOne (coordData [c].count);
          a [i].c [c] = coordData [c].alt [randAlt].value;
        }
      }

      // Bounds checking and discretization
      a [i].c [c] = u.SeInDiSp (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }
}
//————————————————————————————————————————————————————————————————————
The Revision method is used to update information about the best and worst solutions in the current population. The index of the best solution is specified, and the variable storing the best fitness is assigned the value of the worst of the current one. This creates an initial status for searching for new extremes. All solutions in the current population are searched: the solution with the maximum fitness value (the best solution) is determined. Its value is saved and the stored index is updated. We also search for the solution with the minimum fitness value (the worst solution). If a truly best solution has been found, its coordinates are copied into a special array or variable designed to store the current best solution.

As a result of the method execution, the coordinates of the best and worst solutions in the population are known exactly at the current moment in time. This allows the algorithm to track the search dynamics and use this data in decision making to better find optimal solutions.

//————————————————————————————————————————————————————————————————————
//--- Update the best solution
void C_AO_DEA::Revision ()
{
  int bestIND = -1;
  fW = fB;

  // Find the best and worst solutions in the current population
  for (int i = 0; i < popSize; i++)
  {
    // Update the best solution
    if (a [i].f > fB)
    {
      fB = a [i].f;
      bestIND = i;
    }

    // Update the worst solution
    if (a [i].f < fW)
    {
      fW = a [i].f;
    }
  }

  // Copy the coordinates of the best solution
  if (bestIND != -1)
  {
    ArrayCopy (cB, a [bestIND].c, 0, 0, WHOLE_ARRAY);
  }
}
//————————————————————————————————————————————————————————————————————


Test results
Now let's look at the results. It can be immediately noted that the algorithm copes well and fast with various types of tasks.

DEA|Dolphin Echolocation Algorithm|100.0|2.0|2.0|1.0|

=============================
5 Hilly's; Func runs: 10000; result: 0.7599517883429889
25 Hilly's; Func runs: 10000; result: 0.6757192867862007
500 Hilly's; Func runs: 10000; result: 0.34170057553968197
=============================
5 Forest's; Func runs: 10000; result: 0.8958173952258406
25 Forest's; Func runs: 10000; result: 0.6422393144820473
500 Forest's; Func runs: 10000; result: 0.23940903266305935
=============================
5 Megacity's; Func runs: 10000; result: 0.6153846153846154
25 Megacity's; Func runs: 10000; result: 0.4403076923076923
500 Megacity's; Func runs: 10000; result: 0.15115384615384736
=============================
All score: 4.76168 (52.91%)
The visualization shows a spread of results for low-dimensional functions (green), and good results for medium-dimensional functions (blue).

Hilly

DEA on the Hilly test function

Forest

DEA on the Forest test function

Megacity

DEA on the Megacity test function

Based on its results, the DEA algorithm occupies 25th place in the ranking table.

#	AO	Description	Hilly	Hilly
Final	Forest	Forest
Final	Megacity (discrete)	Megacity
Final	Final
Result	% of
MAX
10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)
1	ANS	across neighbourhood search	0.94948	0.84776	0.43857	2.23581	1.00000	0.92334	0.39988	2.32323	0.70923	0.63477	0.23091	1.57491	6.134	68.15
2	CLA	code lock algorithm (joo)	0.95345	0.87107	0.37590	2.20042	0.98942	0.91709	0.31642	2.22294	0.79692	0.69385	0.19303	1.68380	6.107	67.86
3	AMOm	animal migration ptimization M	0.90358	0.84317	0.46284	2.20959	0.99001	0.92436	0.46598	2.38034	0.56769	0.59132	0.23773	1.39675	5.987	66.52
4	(P+O)ES	(P+O) evolution strategies	0.92256	0.88101	0.40021	2.20379	0.97750	0.87490	0.31945	2.17185	0.67385	0.62985	0.18634	1.49003	5.866	65.17
5	CTA	comet tail algorithm (joo)	0.95346	0.86319	0.27770	2.09435	0.99794	0.85740	0.33949	2.19484	0.88769	0.56431	0.10512	1.55712	5.846	64.96
6	TETA	time evolution travel algorithm (joo)	0.91362	0.82349	0.31990	2.05701	0.97096	0.89532	0.29324	2.15952	0.73462	0.68569	0.16021	1.58052	5.797	64.41
7	SDSm	stochastic diffusion search M	0.93066	0.85445	0.39476	2.17988	0.99983	0.89244	0.19619	2.08846	0.72333	0.61100	0.10670	1.44103	5.709	63.44
8	BOAm	billiards optimization algorithm M	0.95757	0.82599	0.25235	2.03590	1.00000	0.90036	0.30502	2.20538	0.73538	0.52523	0.09563	1.35625	5.598	62.19
9	AAm	archery algorithm M	0.91744	0.70876	0.42160	2.04780	0.92527	0.75802	0.35328	2.03657	0.67385	0.55200	0.23738	1.46323	5.548	61.64
10	ESG	evolution of social groups (joo)	0.99906	0.79654	0.35056	2.14616	1.00000	0.82863	0.13102	1.95965	0.82333	0.55300	0.04725	1.42358	5.529	61.44
11	SIA	simulated isotropic annealing (joo)	0.95784	0.84264	0.41465	2.21513	0.98239	0.79586	0.20507	1.98332	0.68667	0.49300	0.09053	1.27020	5.469	60.76
12	BBO	biogeography based optimization	0.94912	0.69456	0.35031	1.99399	0.93820	0.67365	0.25682	1.86867	0.74615	0.48277	0.17369	1.40261	5.265	58.50
13	ACS	artificial cooperative search	0.75547	0.74744	0.30407	1.80698	1.00000	0.88861	0.22413	2.11274	0.69077	0.48185	0.13322	1.30583	5.226	58.06
14	DA	dialectical algorithm	0.86183	0.70033	0.33724	1.89940	0.98163	0.72772	0.28718	1.99653	0.70308	0.45292	0.16367	1.31967	5.216	57.95
15	BHAm	black hole algorithm M	0.75236	0.76675	0.34583	1.86493	0.93593	0.80152	0.27177	2.00923	0.65077	0.51646	0.15472	1.32195	5.196	57.73
16	ASO	anarchy society optimization	0.84872	0.74646	0.31465	1.90983	0.96148	0.79150	0.23803	1.99101	0.57077	0.54062	0.16614	1.27752	5.178	57.54
17	RFO	royal flush optimization (joo)	0.83361	0.73742	0.34629	1.91733	0.89424	0.73824	0.24098	1.87346	0.63154	0.50292	0.16421	1.29867	5.089	56.55
18	AOSm	atomic orbital search M	0.80232	0.70449	0.31021	1.81702	0.85660	0.69451	0.21996	1.77107	0.74615	0.52862	0.14358	1.41835	5.006	55.63
19	TSEA	turtle shell evolution algorithm (joo)	0.96798	0.64480	0.29672	1.90949	0.99449	0.61981	0.22708	1.84139	0.69077	0.42646	0.13598	1.25322	5.004	55.60
20	DE	differential evolution	0.95044	0.61674	0.30308	1.87026	0.95317	0.78896	0.16652	1.90865	0.78667	0.36033	0.02953	1.17653	4.955	55.06
21	SRA	successful restaurateur algorithm (joo)	0.96883	0.63455	0.29217	1.89555	0.94637	0.55506	0.19124	1.69267	0.74923	0.44031	0.12526	1.31480	4.903	54.48
22	CRO	chemical reaction optimization	0.94629	0.66112	0.29853	1.90593	0.87906	0.58422	0.21146	1.67473	0.75846	0.42646	0.12686	1.31178	4.892	54.36
23	BIO	blood inheritance optimization (joo)	0.81568	0.65336	0.30877	1.77781	0.89937	0.65319	0.21760	1.77016	0.67846	0.47631	0.13902	1.29378	4.842	53.80
24	BSA	bird swarm algorithm	0.89306	0.64900	0.26250	1.80455	0.92420	0.71121	0.24939	1.88479	0.69385	0.32615	0.10012	1.12012	4.809	53.44
25	DEA	dolphin_echolocation_algorithm	0.75995	0.67572	0.34171	1.77738	0.89582	0.64223	0.23941	1.77746	0.61538	0.44031	0.15115	1.20684	4.762	52.91
26	HS	harmony search	0.86509	0.68782	0.32527	1.87818	0.99999	0.68002	0.09590	1.77592	0.62000	0.42267	0.05458	1.09725	4.751	52.79
27	SSG	saplings sowing and growing	0.77839	0.64925	0.39543	1.82308	0.85973	0.62467	0.17429	1.65869	0.64667	0.44133	0.10598	1.19398	4.676	51.95
28	BCOm	bacterial chemotaxis optimization M	0.75953	0.62268	0.31483	1.69704	0.89378	0.61339	0.22542	1.73259	0.65385	0.42092	0.14435	1.21912	4.649	51.65
29	ABO	african buffalo optimization	0.83337	0.62247	0.29964	1.75548	0.92170	0.58618	0.19723	1.70511	0.61000	0.43154	0.13225	1.17378	4.634	51.49
30	(PO)ES	(PO) evolution strategies	0.79025	0.62647	0.42935	1.84606	0.87616	0.60943	0.19591	1.68151	0.59000	0.37933	0.11322	1.08255	4.610	51.22
31	FBA	Fractal-Based Algorithm	0.79000	0.65134	0.28965	1.73099	0.87158	0.56823	0.18877	1.62858	0.61077	0.46062	0.12398	1.19537	4.555	50.61
32	TSm	tabu search M	0.87795	0.61431	0.29104	1.78330	0.92885	0.51844	0.19054	1.63783	0.61077	0.38215	0.12157	1.11449	4.536	50.40
33	BSO	brain storm optimization	0.93736	0.57616	0.29688	1.81041	0.93131	0.55866	0.23537	1.72534	0.55231	0.29077	0.11914	0.96222	4.498	49.98
34	WOAm	wale optimization algorithm M	0.84521	0.56298	0.26263	1.67081	0.93100	0.52278	0.16365	1.61743	0.66308	0.41138	0.11357	1.18803	4.476	49.74
35	AEFA	artificial electric field algorithm	0.87700	0.61753	0.25235	1.74688	0.92729	0.72698	0.18064	1.83490	0.66615	0.11631	0.09508	0.87754	4.459	49.55
36	AEO	artificial ecosystem-based optimization algorithm	0.91380	0.46713	0.26470	1.64563	0.90223	0.43705	0.21400	1.55327	0.66154	0.30800	0.28563	1.25517	4.454	49.49
37	CAm	camel algorithm M	0.78684	0.56042	0.35133	1.69859	0.82772	0.56041	0.24336	1.63149	0.64846	0.33092	0.13418	1.11356	4.444	49.37
38	ACOm	ant colony optimization M	0.88190	0.66127	0.30377	1.84693	0.85873	0.58680	0.15051	1.59604	0.59667	0.37333	0.02472	0.99472	4.438	49.31
39	CMAES	covariance_matrix_adaptation_evolution_strategy	0.76258	0.72089	0.00000	1.48347	0.82056	0.79616	0.00000	1.61672	0.75846	0.49077	0.00000	1.24923	4.349	48.33
40	BFO-GA	bacterial foraging optimization - ga	0.89150	0.55111	0.31529	1.75790	0.96982	0.39612	0.06305	1.42899	0.72667	0.27500	0.03525	1.03692	4.224	46.93
41	SOA	simple optimization algorithm	0.91520	0.46976	0.27089	1.65585	0.89675	0.37401	0.16984	1.44060	0.69538	0.28031	0.10852	1.08422	4.181	46.45
42	ABHA	artificial bee hive algorithm	0.84131	0.54227	0.26304	1.64663	0.87858	0.47779	0.17181	1.52818	0.50923	0.33877	0.10397	0.95197	4.127	45.85
43	ACMO	atmospheric cloud model optimization	0.90321	0.48546	0.30403	1.69270	0.80268	0.37857	0.19178	1.37303	0.62308	0.24400	0.10795	0.97503	4.041	44.90
44	ADAMm	adaptive moment estimation M	0.88635	0.44766	0.26613	1.60014	0.84497	0.38493	0.16889	1.39880	0.66154	0.27046	0.10594	1.03794	4.037	44.85
45	CGO	chaos game optimization	0.57256	0.37158	0.32018	1.26432	0.61176	0.61931	0.62161	1.85267	0.37538	0.21923	0.19028	0.78490	3.902	43.35
RW	random walk	0.48754	0.32159	0.25781	1.06694	0.37554	0.21944	0.15877	0.75375	0.27969	0.14917	0.09847	0.52734	2.348	26.09


Summary
The main advantages of the algorithm are: adaptive search control: the predetermined probability gradually increases, shifting the balance from exploration to exploitation, dynamic adaptation, collective memory: accumulated fitness preserves and disseminates information about promising areas, diversity mechanism: resetting information for the best positions stimulates exploration of new areas.

The strength of the algorithm lies in its adaptive nature. The dynamic coefficient makes the algorithm responsive to the population state and adapts to the rhythm of the task. When the dolphins are scattered across the search space, the algorithm encourages local exploration. When they begin to converge on a goal, it pushes them to new horizons, preventing them from getting stuck in the illusion of local success.

Accumulated fitness is the collective memory of the flock, where every discovery leaves an echo in the solution space. But unlike simple accumulation, the algorithm can forget — zeroing out the best positions paradoxically leads to even better discoveries.

This is not just a metaphor - it is a philosophy of optimization, where every click of the echolocator carries information about the space of possibilities.

READ  7
Detecting and Classifying Fractal Patterns Using Machine Learning
MetaTrader 5 — Trading | 29 May 2026 at 15:06
1 487
11
dmitrievsky
[Deleted]
Introduction
In the first article, we have examined in detail the fundamental aspects of the multifractal market theory. We noted that price charts are capable of forming certain repeating structures under the influence of external information that organizes them. Market participants create a complex dynamic system that has memory elements that take the form of certain market symmetries (patterns). These patterns may evolve over time or may repeat. Due to the self-similarity of fractal market structures, patterns can be expressed across different time scales.

This article presents an original approach to identifying and classifying fractal patterns. The analysis will be conducted in Python, with the ability to export final models to the MetaTrader 5 terminal in ONNX format.

Before you begin, make sure you have installed all the required packages and modules. Some of the imported modules are included in the attachment below.

import pandas as pd
import math
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from bots.botlibs.labeling_lib import *
from bots.botlibs.tester_lib import test_model
from bots.botlibs.export_lib import export_model_to_ONNX


Implementation of the fractal pattern search function
In this article, I propose a simple approach to finding symmetrical multifractal market structures through correlation. We can explore both fractal and multifractal patterns, which are scale invariant, that is, they have different sizes. To do this, it is necessary to implement a search for patterns through correlation on different time scales, which will be specified in the settings. Below is a function that calculates correlation in a sliding window taking into account the variable length of patterns.

@njit
def calculate_symmetric_correlation_dynamic(data, min_window_size, max_window_size):
    n = len(data)
    min_w = max(2, min_window_size)
    max_w = max(min_w, max_window_size)
    num_correlations = max(0, n - min_w + 1)

    if num_correlations == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    correlations = np.zeros(num_correlations, dtype=np.float64)
    best_window_sizes = np.full(num_correlations, -1, dtype=np.int64)

    for i in range(num_correlations):
        max_abs_corr_for_i = -1.0
        best_corr_for_i = 0.0
        current_best_w = -1
        current_max_w = min(max_w, n - i)
        start_w = min_w
        if start_w % 2 != 0:
            start_w += 1

        for w in range(start_w, current_max_w + 1, 2):
            if w < 2 or i + w > n:
                continue
            half_window = w // 2
            window = data[i : i + w]
            first_half = window[:half_window]
            second_half = (window[half_window:] * -1)[::-1]
            
            std1 = np.std(first_half)
            std2 = np.std(second_half)

            if std1 > 1e-9 and std2 > 1e-9:
                mean1 = np.mean(first_half)
                mean2 = np.mean(second_half)
                cov = np.mean((first_half - mean1) * (second_half - mean2))
                corr = cov / (std1 * std2)
                if  abs(corr) > max_abs_corr_for_i:
                    max_abs_corr_for_i = abs(corr)
                    best_corr_for_i =corr
                    current_best_w = w
        
        correlations[i] = best_corr_for_i
        best_window_sizes[i] = current_best_w
        
    return correlations, best_window_sizes

To speed up a loop with similar calculations (loops are slow in Python), the @njit decorator is used, which speeds up calculations using the Numba package. 

The function accepts as input a data frame with closing prices, as well as the minimum and maximum "window" sizes for patterns. For example, we want to calculate the correlation for patterns whose length is from 100 to 200 bars. Then we set the appropriate settings, after which the correlation between its left and mirror-inverted right parts is checked for each new reference point and for each given pattern length. The inversion of the right half is highlighted in yellow. This is very important because we are looking for symmetry in the data.

The values of the best absolute correlations for each starting point are written to the correlations[] array. The window size (pattern length) corresponding to the best correlation is written to another array best_window_sizes[]. Thus, the function returns the maximum correlation values and the corresponding pattern for each starting point.



Visual check of the found patterns
Once all the patterns have been calculated, we can visually assess the correctness of our algorithm. For this purpose, I propose another function that will display the top best patterns found by the highest absolute Pearson correlation coefficient.

def plot_best_n_patterns(data, min_window_size, max_window_size, n_best):
    # 1. Calculate correlations and best window sizes
    corrs, window_sizes = calculate_symmetric_correlation_dynamic(data, min_window_size, max_window_size)

    # 2. Find N best patterns
    # Assuming -1 in window_sizes means invalid/not found by the calculation logic
    valid_calc_mask = window_sizes != -1 
    
    if not np.any(valid_calc_mask):
        print("No suitable patterns found (all window sizes were marked as -1 by calculation).")
        return
        
    filtered_corrs = corrs[valid_calc_mask]
    filtered_window_sizes = window_sizes[valid_calc_mask]
    
    original_indices_all = np.arange(len(corrs)) 
    filtered_start_indices = original_indices_all[valid_calc_mask]

    if len(filtered_corrs) == 0: 
        print("No suitable patterns found after filtering out -1 window_sizes.")
        return

    # Sort by absolute correlation value in descending order
    sorted_indices_of_filtered = np.argsort(np.abs(filtered_corrs))[::-1]
    
    # Determine how many of the top patterns to consider
    num_to_consider = min(n_best, len(sorted_indices_of_filtered))

    if num_to_consider == 0:
        print("No patterns to plot (either n_best is too small, or no patterns passed the initial filter).")
        return

    # Pre-filter these top candidates to find those actually plottable (even window size >= 2)
    patterns_to_plot_details = []
    for i in range(num_to_consider):
        idx_in_filtered_arrays = sorted_indices_of_filtered[i] # Index within the already filtered (by valid_calc_mask) arrays
        
        w_best_candidate = filtered_window_sizes[idx_in_filtered_arrays]
        actual_data_start_index = filtered_start_indices[idx_in_filtered_arrays]
        correlation_value = filtered_corrs[idx_in_filtered_arrays]
        
        # Check if the window size is valid for plotting (even and sufficiently large)
        if w_best_candidate >= 2 and w_best_candidate % 2 == 0 : 
            patterns_to_plot_details.append({
                "original_rank_in_consider_list": i, # Rank among the num_to_consider items
                "data_start_index": actual_data_start_index,
                "correlation": correlation_value,
                "window_size": int(w_best_candidate) # Ensure it's int
            })
        else:
            print(f"Info: Top candidate (originally rank {i+1} among considered, "
                  f"Start Index: {actual_data_start_index}) "
                  f"skipped due to invalid window size for plotting: {w_best_candidate} (must be even and >= 2).")


    num_actually_plotted = len(patterns_to_plot_details)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) # Single axes for combined plot
    title_fontsize = 12
    label_fontsize = 10
    legend_fontsize = 8
    tick_labelsize = 9

    if num_actually_plotted == 0:
        # This message is shown if, out of the top 'num_to_consider' patterns, none had a valid window size for plotting.
        print("No patterns with valid window sizes (even, >=2) found among the top candidates to display on the chart.")
        ax.text(0.5, 0.5, "No valid patterns to display on the chart.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=title_fontsize, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle(f"Symmetric Patterns Overlaid", fontsize=title_fontsize) # Generic title
    else:
        # Generate distinct colors for each pattern that will actually be plotted
        plot_colors = plt.cm.viridis(np.linspace(0, 1, num_actually_plotted))

        for plot_idx, pattern_info in enumerate(patterns_to_plot_details):
            actual_data_start_index = pattern_info["data_start_index"]
            correlation_value = pattern_info["correlation"]
            w_best = pattern_info["window_size"]
            
            half_window = w_best // 2
            
            # Ensure indices are within data bounds
            if actual_data_start_index + w_best > len(data):
                print(f"Warning: Pattern P{plot_idx+1} (Idx:{actual_data_start_index}, W:{w_best}) extends beyond data length {len(data)}. Skipping.")
                continue

            left_part_data = data[actual_data_start_index : actual_data_start_index + half_window]
            right_part_data = data[actual_data_start_index + half_window : actual_data_start_index + w_best]
            
            x_indices = np.arange(w_best) # X-axis relative to pattern start
            current_color = plot_colors[plot_idx]

            # Plot left part
            ax.plot(x_indices[:half_window], left_part_data, 
                    color=current_color, linestyle='-', 
                    label=f"P{plot_idx+1} (Idx:{actual_data_start_index}, W:{w_best}, C:{correlation_value:.2f})")
            
            # Plot right part
            ax.plot(x_indices[half_window:], right_part_data, 
                    color=current_color, linestyle='--') 
            
            # Add a vertical line to mark the split point for this pattern
            ax.axvline(x=half_window - 0.5, color=current_color, linestyle=':', linewidth=1, alpha=0.6)

        ax.set_xlabel("Index within Pattern Window", fontsize=label_fontsize)
        ax.set_ylabel("Data Value", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax.grid(True)
        
        ax.legend(fontsize=legend_fontsize, loc='best')
        # Add a text note to explain line styles
        fig.text(0.99, 0.01, 'Solid: Left Part, Dashed: Right Part (Original)', 
                 horizontalalignment='right', verticalalignment='bottom', 
                 fontsize=legend_fontsize - 1, color='dimgray', transform=fig.transFigure)

        fig.suptitle(f"Top {num_actually_plotted} Symmetric Patterns Overlaid", fontsize=title_fontsize)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect for suptitle and fig.text
    plt.show()


This function is fairly lengthy, but most of the code handles pattern sorting and plotting. First, the patterns themselves are calculated, then they are sorted by the values of the correlation coefficient. Each pattern’s position in the price history is identified and then plotted. The result of this function is shown below.

In the first image, we see one selected pattern that has the highest absolute correlation. It resembles a certain peak, local or global, which symbolizes a trend reversal. The dotted vertical line marks the split of the series into two equal halves. The right half is sign-inverted and mirrored. After this, the correlation between the left and right sections is calculated. The charts do not show the inverted right-hand sides, but the original source series of quotes.



Fig. 1. The best pattern has a period of 50 and a correlation of -0.98

In the second figure, I have displayed the five best patterns with a period of 50. Of these five patterns, three resemble tops, two resemble bottoms; one of them also looks like a continuation of a bullish trend. The scale on the left displays the historical price levels these patterns correspond to.



Fig. 2. Top five 50-period patterns

If we increase the pattern periods to 150 bars, completely different structures are observed. Three similar patterns were found (top). This is because a small shift in history resulted in the discovery of the same structure. The other two patterns turned out different from each other.



Fig. 3. Top five patterns with a period of 150

If we increase the pattern calculation window to 250, the same patterns again turned out to be among the best, but with a slight shift in history. Some reversal patterns can also be observed, as their correlations are negative.



Fig. 4. Top five patterns with a period of 250

These illustrations demonstrate a wide variety of self-affine (self-similar) market structures. Theoretically, this diversity can only be limited by the length of the series under study. In this case, it is quite difficult to determine which specific pattern will have predictive potential and which will not. It would take months to study each individual structure. Machine learning can help us here, as it allows us to classify all patterns at once.

It is quite possible that searching for structures through correlation is not ideal, and other, more accurate estimation methods should be considered. But this approach is a good starting point for further research and is intuitive. Now we need to figure out how to analyze these market fractals and build a trading system based on them using machine learning.



Labeling trades based on symmetrical structures
The function of finding symmetric structures is, in a sense, a data mining function. We set clear criteria for what we are looking for in the data - self-similar fractal structures. Next, it is necessary to collect and classify the information received. But even this will not be enough, because we will have to come up with a way to label trades based on this data, which is what we will do in this section.

I propose the following method of labeling trades for subsequent classification. It is not the only possible one, but reflects the author’s understanding of how it can be implemented. I believe that further research is needed on this topic, but for now we will limit ourselves to the existing markup method.

@njit
def generate_future_outcome_labels_for_patterns(
    close_data_len,                 #  Total length of the original close_data
    correlations_at_window_start,   # Correlation array
    window_sizes_at_window_start,   # Array of window sizes
    source_close_data,              # Full close_data array
    correlation_threshold,
    min_future_horizon,             # Minimum horizon for determining the future price
    max_future_horizon,             # Maximum horizon
    markup_points                   # "Markup" for determining a significant price change
):
    labels = np.full(close_data_len, 2.0, dtype=np.float64)  # 2.0: no signal/neutral/no pattern
    num_potential_windows = len(correlations_at_window_start)

    for idx_window_start in range(num_potential_windows):
        corr_value = correlations_at_window_start[idx_window_start]
        w = window_sizes_at_window_start[idx_window_start]

        # Condition 1: The correlation should be strong enough
        if abs(corr_value) < correlation_threshold:
            continue

        # Condition 2: A valid window should be found
        if w < 2:
            continue

        # The point in time (index) when the correlation pattern is fully formed
        signal_time_idx = idx_window_start + w - 1

        if signal_time_idx >= close_data_len: # Theoretically, this should not happen
            continue
            
        # Array for storing labels for the entire pattern (both left and right parts)
        pattern_labels = []
            
        # Calculate individual marks for all points of the pattern
        for point_idx in range(idx_window_start, signal_time_idx + 1):
            # Current price for this particular point
            current_price = source_close_data[point_idx]
            
            # Define the forecast horizon
            current_horizon = min_future_horizon
            if max_future_horizon > min_future_horizon:
                current_horizon = random.randint(min_future_horizon, max_future_horizon)
            
            # Index of future price relative to the current point
            future_price_idx = point_idx + current_horizon
            
            if future_price_idx >= close_data_len:
                continue
                
            future_price = source_close_data[future_price_idx]
            
            # Define a label for the current point
            current_label = 2.0  # Neutral by default
            if future_price > current_price + markup_points:
                current_label = 0.0  # Price increased
            elif future_price < current_price - markup_points:
                current_label = 1.0  # Price fell
                
            # Add the label to the array if it is not neutral
            if current_label != 2.0:
                pattern_labels.append(current_label)
        
        # If there are no significant marks in the pattern, move on to the next pattern
        if len(pattern_labels) == 0:
            continue
            
        # Calculate the average mark for all points of the pattern
        avg_label = 0.0
        for l in pattern_labels:
            avg_label += l
        avg_label /= len(pattern_labels)
        
        # Define a common label for the entire pattern
        pattern_label = 0.0 if avg_label < 0.5 else 1.0
        
        # Assign this label to all points of the pattern
        for i in range(idx_window_start, signal_time_idx + 1):
            labels[i] = pattern_label
        
    return labels

The generate_future_outcome_labels_for_patterns() function implements the following functionality:

The input is the original array of prices, an array of correlations, and an array of pattern lengths corresponding to the largest correlations for a particular data point. The function also accepts a minimum and maximum forecast horizon, in bars.
Initially, all trades are marked as 2.0 (do not trade).
The loop checks the correlation value for each point in the time series. If the correlation exceeds correlation_threshhold, then this observation undergoes additional processing, otherwise the label for this example remains 2.0.
Then, throughout the entire length of the pattern, determined through the maximum correlation, trades are calculated based on future price changes. For each point: if the price has risen, then this is the 0 mark - buy, if the price has fallen, then this is 1 - sell. The average value is found across trades, and an average mark is assigned to each observation of the current pattern.
The philosophy behind this approach is that highly correlated structures have a "memory" of their initial conditions and exhibit some degree of regularity. This means that observations within them are better predicted, but to prevent overfitting, we put an average label on each observation. Conversely, observations within low-correlation structures are poorly predicted because they have less regularity.

As a result, we exploit the following principle: one model will determine the quality of the pattern (whether it is worth trading at the moment or not), and the other model will determine the direction of trading. Machine learning will be tasked with approximating all possible patterns and trading directions.

Next, we will need another orchestrator function, which will be called directly for labeling trades.



The final fractal pattern-based markup function
It is time to put it all together and write a trade labeling tool that is ready to use. 

def get_fractal_pattern_labels_from_future_outcome(
    dataset,
    min_window_size=6,
    max_window_size=60,
    correlation_threshold=0.7,
    min_future_horizon=5, 
    max_future_horizon=5,    
    markup_points=0.00010,  
):
    if 'close' not in dataset.columns:
        raise ValueError("Dataset must contain a 'close' column.")

    close_data = dataset['close'].values
    n_data = len(close_data)

    if min_window_size < 2:
        min_window_size = 2
    if max_window_size < min_window_size:
        max_window_size = min_window_size
    if min_future_horizon <= 0:
        raise ValueError("min_future_horizon must be > 0")
    if max_future_horizon < min_future_horizon:
        raise ValueError("max_future_horizon must be >= min_future_horizon")
    
    correlations_at_start, best_window_sizes_at_start = calculate_symmetric_correlation_dynamic(
        close_data,
        min_window_size,
        max_window_size,
    )

    labels = generate_future_outcome_labels_for_patterns(
        n_data,
        correlations_at_start,
        best_window_sizes_at_start,
        close_data,
        correlation_threshold,
        min_future_horizon,
        max_future_horizon,
        markup_points
    )

    result_df = dataset.copy()
    result_df['labels'] = pd.Series(labels, index=dataset.index)    
    return result_df
The get_fractal_pattern_labels_from_future_outcome() function is called directly to label your dataset:

the input is a data frame, which should contain a "close" column with closing prices, as well as features (optional);
the minimum and maximum length of patterns that will be used in marking trades is set;
a correlation threshold is set, which allows us to adjust the "accuracy" of the patterns involved in the marking;
the minimum and maximum position holding time (in bars) for marking trades should also be specified;
optionally, we can set markup.
The function takes a dataset of closing prices and labels trades based on fractal patterns, adding a "labels" column with the labeled marks.



Training a machine learning model based on fractal markup
Now everything is ready for experiments and you can train the models. As source data, I took hourly EURUSD quotes from 2010 to present.

It was decided to use standard deviations in sliding windows of different periods as features:

def get_features(data: pd.DataFrame) -> pd.DataFrame:
    pFixed = data.copy()
    pFixedC = data.copy()
    count = 0

    for i in hyper_params['periods']:
        pFixed[str(count)] = pFixedC.rolling(i).std()
        count += 1

    return pFixed.dropna()
Next, you need to correctly set the model hyperparameters:

# set hyper parameters
hyper_params = {
    'symbol': 'EURUSD_H1',
    'export_path': '/Users/dmitrievsky/Library//drive_c/Program Files/MetaTrader 5/MQL5/Include/Trend following/',
    'model_number': 0,
    'markup': 0.00010,
    'stop_loss':  0.00500,
    'take_profit': 0.00500,
    'periods': [i for i in range(15, 300, 30)],
    'backward': datetime(2010, 1, 1),
    'forward': datetime(2024, 1, 1),
}
stop loss and take profit are the same and equal to 500 five-digit points;
next, we need to specify the path to export trained models to our folder;
we will set the periods of features (standard deviations) in the range from 15 to 300, with a step of 30 (there are 10 features in total);
the training period is from 2010 to 2024, the rest of data remains outside the training.
The main training loop allows training multiple models at once, and it also allows iterating over hyperparameters:

# fit the models
models = []
for i in range(10):
    print('Learn ' + str(i) + ' model')
    dataset = get_features(get_prices())
    data = dataset[(dataset.index < hyper_params['forward']) & (dataset.index > hyper_params['backward'])].copy()
    data = get_fractal_pattern_labels_from_future_outcome(data, 100, 100, 0.9, 15, 25, 0.00010)
    models.append(fit_final_models(data))
In the loop, we first obtain prices and features, then determine the timeframe on which the model will be trained.

In the get_fractal_pattern_labels_from_future_outcome() function, pass the following parameters:

original dataframe with prices and features
minimum window for calculating correlation
maximum window for calculating correlation
correlation coefficient threshold for patterns, default 0.9
minimum forecast horizon in bars
maximum forecast horizon in bars
markup in points
The labeled data is then fed into a function that trains two classifiers:

def fit_final_models(dataset: pd.DataFrame) -> list:
    feature_columns = dataset.columns[1:-1]

    # 1. Data for the main model
    # Filter the dataset: only those examples where 'labels' are equal to 0 or 1 are used for the main model.
    main_model_df = dataset[dataset['labels'].isin([0, 1])].copy()
    
    X = main_model_df[feature_columns]
    y = main_model_df['labels'].astype('int16')

    # 2. Data for the meta model
    X_meta = dataset[feature_columns]
    
    # Modify labels for the meta model: if 'labels' contains 1 or 0, then the new label is 1, if 2, then 0.
    y_meta = dataset['labels'].apply(lambda label_val: 1 if label_val in [0, 1] else 0).astype('int16')

    # For the main model
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.7, test_size=0.3, shuffle=True) 
    
    # For the meta model
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.7, test_size=0.3, shuffle=True)
    
    # Train the main model
    model = CatBoostClassifier(iterations=1000,
                               custom_loss=['Accuracy'],
                               eval_metric='Accuracy',
                               verbose=False,
                               use_best_model=True,
                               task_type='CPU',
                               )
    
    # Check if the samples are empty after splitting (unlikely if X is large enough)
    if not train_X.empty and not test_X.empty:
        model.fit(train_X, train_y, eval_set=(test_X, test_y),
                  early_stopping_rounds=25, plot=False)
    elif not train_X.empty: # If the test sample is empty, but the training sample exists
        print("Warning: The test sample (test_X) for the main model is empty. The model is trained without eval_set.")
        model.fit(train_X, train_y, early_stopping_rounds=15, plot=False) # use_best_model may not work correctly without eval_set
    else: # If the training set is empty
        print("Error: The training set (train_X) for the main model is empty. The model cannot be trained.")
        # In this case, test_model will most likely throw an error later.
        # Return R2=-1 and the untrained model, the meta model will also not make sense without the main one.
        print("R2 is fixed at -1.0, models are not trained.")
        return [-1.0, model, None] # model - instance, but not trained

    # Meta model training
    meta_model = CatBoostClassifier(iterations=1000,
                                    custom_loss=['F1'],
                                    eval_metric='F1',
                                    verbose=False,
                                    use_best_model=True,
                                    task_type='CPU',
                                    )

    if not train_X_m.empty and not test_X_m.empty:
        meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m),
                       early_stopping_rounds=25, plot=False)
    elif not train_X_m.empty:
        print("Warning: The test sample (test_X_m) for the meta model is empty. The meta model is trained without eval_set.")
        meta_model.fit(train_X_m, train_y_m, early_stopping_rounds=25, plot=False)
    else:
        print("Error: The training set (train_X_m) for the meta model is empty. The meta model cannot be trained.")
        print("R2 fixed as -1.0.")
        return [-1.0, model, meta_model] # meta_model - instance, but not trained

    data_for_test = get_features(get_prices())
    R2 = test_model(data_for_test, 
                    [model, meta_model], 
                    hyper_params['stop_loss'], 
                    hyper_params['take_profit'],
                    hyper_params['forward'],
                    hyper_params['backward'],
                    hyper_params['markup'],
                    plt=False)
    
    if math.isnan(R2):
        R2 = -1.0
        print('R2 fixed as -1.0')
    print('R2: ' + str(R2))
    result = [R2, model, meta_model]
    return result

Points that deserve special attention are highlighted in bold. The main model is trained to predict only 0 or 1 labels, while the additional meta-model predicts whether to trade or not.



Testing and final results
To begin with, it is worth saying that I tested the algorithm only on EURUSD. I was able to select a pattern window size that works best on new data. It equals 100. The optimal parameters of the algorithm are already specified in the code, so you can reproduce the result yourself.

The balance graph for training and test data looks like this:



Fig. 5. Testing an algorithm based on fractal markup

There is a direct relationship between the correlation threshold and trading results on new data. For example, for a threshold of 0.7, the balance graph already indicates clear overfitting. This reflects the fact that weak correlation between two parts of a time series leads to weak dependence. Weak dependence, in turn, does not allow for the correct classification of reliable patterns, because they are mixed with unreliable ones.



Fig. 6. Testing the algorithm with a threshold of 0.7

It seems that correct pattern identification is critical. Further research and insights are needed on how best to organize the search for fractal structures.

The quality and quantity of features also influence the classification results. If we use increments instead of standard deviations, the balance chart will look different.

It is also necessary to analyze and reasonably criticize the method of labeling trades based on the patterns found.

Error analysis of CatBoost models shows that the models are trained with low error:

>>> models[-1][1].get_best_score()['validation']
{'Accuracy': 0.9700523560209424, 'Logloss': 0.17002244404784328}
>>> models[-1][2].get_best_score()['validation']
{'Logloss': 0.25629795409043277, 'F1': 0.8455473098330242}
>>> 


Exporting and testing models in the MetaTrader 5 terminal
To export models, we need to call the function:

export_model_to_ONNX(model = models[-1],
                     symbol = hyper_params['symbol'],
                     periods = hyper_params['periods'],
                     periods_meta = hyper_params['periods'],
                     model_number = hyper_params['model_number'],
                     export_path = hyper_params['export_path'])
After exporting and compiling the EA, we obtained the following results:



Fig. 7. Testing the EA for the entire period



Fig. 8. Testing the EA on new data


Conclusion
In this article, we have touched upon the intriguing topic of fractal analysis and market forecasting using machine learning. These are just the first steps towards exploring the diverse fractal structures that form on financial price charts.

It should be noted that correlation searches may not fully reflect the relationships between past and future price series, and this topic requires further research. For example, regression analysis may be more appropriate than correlation analysis. At the same time, the current algorithm is capable of demonstrating good predictive capabilities when properly configured, which confirms the presence of fractal self-similar structures in financial time series.



The Python files.zip archive contains the following files for development in the Python environment:

Filename	Description
fractal patterns.py 
The main script for training models
labeling_lib.py
Updated trade-labeling module
tester_lib.py
Updated custom strategy tester based on machine learning
export_lib.py	Module for exporting models to the terminal
EURUSD_H1.csv
Quote data exported from MetaTrader 5
The MQL5 files.zip archive contains files for the MetaTrader 5 terminal:

Filename	Description
fractal trader.ex5
The compiled bot from the article
fractal trader.mq5
Bot source code from the article
Include//Trend following folder
The ONNX models and t

READS 10

Billiards Optimization Algorithm (BOA)
MetaTrader 5 — Tester | 5 January 2026 at 13:40
3 737
0
Andrey Dik
Andrey Dik
Contents
Introduction
Implementation of the algorithm
Test results


Introduction

In the world of optimization algorithms, where mathematical precision meets real-world inspiration, strikingly ingenious approaches are often born. One such method is the Billiards Optimization Algorithm (BOA), which draws ideas for a search strategy from the mechanics of the classic game of billiards.

Imagine a pool table where each pocket is a potential solution, and the balls are solution seekers moving through the space of possibilities. Just as a skilled billiard player calculates the trajectory of a ball to accurately pocket it, BOA guides its decisions toward optimal results through an iterative process of search and refinement. This algorithm, developed by researchers Hadi Givi and Marie Hubálovská in 2023, demonstrates an interesting and unusual approach to solving optimization problems.

In this article, we dive into the conceptual basis of BOA, explore its mathematical model, and analyze its efficiency in solving multimodal problems. The algorithm, which combines conceptual simplicity with mathematical precision, opens new horizons in the field of computational optimization and represents a valuable addition to the arsenal of modern optimization methods.



Implementation of the algorithm
The BOA algorithm is an optimization method inspired by the game of billiards. Imagine that you are looking for the best solution to some problem, in billiard terminology, it is like trying to get the ball into the pocket. There are 8 pockets on a pool table, as well as many balls. At the beginning of the algorithm, a group of random solutions (population) is created. These decisions are like balls on a pool table. For each solution, the objective function value is calculated to determine how good it is.

At each iteration of the algorithm, the eight best solutions from the population become "pockets" (targets to strive for). The remaining solutions are treated as balls that need to be directed towards these pockets. For each ball (solution), one of the pockets (best solutions) is randomly selected. The new position of the ball is then calculated - a new solution moving in the direction of the chosen pocket. If the new position of the ball gives a better value of the objective function, then the ball moves to the new position, and if not, then it remains in place.

Mathematically it looks like this:

X_new = X + rnd [0.0; 1.0] × (P - I × X)

where:

X_new — new position of the ball,
X — current position of the ball,
P — position of the pocket (or the ball that the current ball is supposed to hit),
I — random selection of numbers 1 or 2.
The process is repeated many times, and eventually the balls (solutions) should approach the optimal solution to the problem.

An advantage of BOA may lie in the fact that it should theoretically balance well, with just one coefficient in the equation, between global search and local search. This is achieved by using a random ratio I, which ensures either an "undershoot" of the ball (refining solutions near good points) or an "overshoot" (exploration of different areas of the solution space).

boa-algorithm-diagram

Figure 1. The BOA algorithm flow chart

Figure 1 schematically shows the operating principle of the BOA algorithm. The central element, a white ball marked "X", represents the agent's current position in the solution search space. Around this ball are colored balls marked "P" - these are "pockets" or "holes" representing the 8 best solutions found so far. The diagram demonstrates how an agent (the white ball) can update its position by moving towards different pockets, namely, for each step, the agent randomly chooses one of the eight pockets as the target direction of movement.

The orange lines with arrows show the possible trajectories of the agent's movement to different pockets (in this case, to the red and blue ones). The dotted circles represent intermediate positions that the agent can take while moving, depending on the value of I (1 or 2). The value of I changes the "strength of the blow" and the nature of the movement: at I=1, the position shifts in the direction of the selected pocket, and at I=2, the agent can make sharper movements, which facilitates the exploration of a larger part of the solution space.

Now that we have fully understood how the algorithm works, we will start writing BOA pseudocode.

Initialization
    Determine the number of balls (popSize) and pockets (numPockets).
    Create a population of balls (agents).
    Set the search space with minimum and maximum boundaries.

Basic algorithm
First stage: Initial initialization (performed only once)
    For each ball in the population:
        Randomly place balls in the search space.
        Save its initial position.
        Set the initial values of the fitness function to the minimum.

Second stage: Moving balls
    For each ball in the population:
        For each coordinate of the ball:
            Choose a random pocket (one of the numPockets best solutions).
            Update the position of the ball using the equation: X_new = X + rnd [0.0; 1.0] × (P - I × X)
            Check that the new position remains within acceptable limits.

Third stage: Update the best solutions
    For each ball:
        If the value of the ball fitness function is better than the global best, update the global best.
        If the value of the ball fitness function is better than its own best, update its best.
    Sort the balls by their best fitness function values.
        The first numPockets of agents after sorting become pockets for the next iteration.

Repeat the Ball movement and Best solution update steps until a stopping condition is reached (for example, the maximum number of iterations).

Let's start writing the algorithm code. The C_AO_BOA class is derived from the C_AO class (the base class of population optimization algorithms) and implements the BOA optimization algorithm. Let's take a look at its key elements:

The C_AO_BOA () constructor initializes the values of instance variables:
popSize is set to 50, which represents the number of balls (agents) in the algorithm.
numPockets is set to 8, it forms the number of pockets on a pool table.
The params array size is changed and two parameters (popSize and numPockets) are added to the params array.
Methods:
SetParams () — the method is responsible for setting the parameters from the 'params' array back into the local popSize and numPockets variables.
Init () — initialization method configures the minimum, maximum values and steps for searching, as well as sets the number of epochs.
Moving () — manages the movement of balls in the algorithm.
Revision () — verifies and revises agents’ positions/decisions.
//——————————————————————————————————————————————————————————————————————————————
class C_AO_BOA : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_BOA () { }
  C_AO_BOA ()
  {
    ao_name = "BOA";
    ao_desc = "Billiards Optimization Algorithm";
    ao_link = "https://www.mql5.com/en/articles/17325";

    popSize    = 50;  // number of balls (agents)
    numPockets = 8;   // number of pockets on a billiard table

    ArrayResize (params, 2);

    params [0].name = "popSize";    params [0].val = popSize;
    params [1].name = "numPockets"; params [1].val = numPockets;
  }

  void SetParams ()
  {
    popSize    = (int)params [0].val;
    numPockets = (int)params [1].val;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  int numPockets;       // number of pockets (best solutions)

  private: //-------------------------------------------------------------------
};
//——————————————————————————————————————————————————————————————————————————————
The Init method in the C_AO_BOA class is responsible for initializing the BOA algorithm. At the beginning of the method, the StandardInit() function is called, passing arrays of minimum and maximum values, as well as steps to it. The function is responsible for performing common actions and initializations that must be performed for all derived classes of optimization algorithms (including initial range setup), as well as preparing the underlying search agents in the algorithm. If StandardInit () returns 'false' (initialization failed), the Init method also returns 'false'. If everything goes well, the method returns 'true'.

//——————————————————————————————————————————————————————————————————————————————
//--- Initialization
bool C_AO_BOA::Init (const double &rangeMinP  [],
                     const double &rangeMaxP  [],
                     const double &rangeStepP [],
                     const int epochsP = 0)
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  return true;
}
//——————————————————————————————————————————————————————————————————————————————
The Moving method implements the main step of the BOA algorithm and controls the movement of agents (balls) in the solution space. First, the if (!revision) condition is checked to determine whether the method is being called for the first time. If revision=false, agent positions need to be initialized. To do this, a loop is executed over all agents defined as popSize, within which a nested loop is executed over the coordinates that define the decision parameters of each agent.

At the stage of generating initial positions, a random value for each agent coordinate is selected in a given range, and the u.SeInDiSp() function brings the value to an acceptable value, taking into account the step. The agent's initial position is stored in a[p].cB[c] as the ball's best individual solution (on the first iteration, the initial solution is equivalent to the best known one), and after setting revision=true, the method terminates, preventing reinitialization of values.

On the second and subsequent iterations, a loop is started for all agents to move. During the coordinate update, nested loops are performed over all agents and their coordinates, where one of the best available pockets is randomly selected to update the agent's current position. The position is updated based on the previous position plus a random change based on the position of the selected pocket. The u.RNDprobab() function returns a random number in the range [0.0; 1.0] to add random noise, while the u.RNDintInRange(1, 2) function multiplies a random value of 1 or 2 with the agent's position.

After updating the position, it is adjusted, bringing the updated value to the specified range, taking into account the minimum and maximum values, as well as the change step.

//——————————————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm
void C_AO_BOA::Moving ()
{
  //----------------------------------------------------------------------------
  // Initial initialization
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        a [p].cB [c] = a [p].c [c];  // Save the initial position
      }
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  for (int p = 0; p < popSize; p++)
  {
    for (int c = 0; c < coords; c++)
    {
      int pocketID = u.RNDminusOne (numPockets);

      a [p].c [c] = a [p].cB [c] + u.RNDprobab () * (a [pocketID].cB [c] - u.RNDintInRange (1, 2) * a [p].cB [c]);
      a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
The Revision method is responsible for updating the best global solution in the BOA algorithm, and also updates the best individual solutions of the balls. At the end of the method, the balls are sorted according to their best individual solutions.
//——————————————————————————————————————————————————————————————————————————————
//--- Update the best solution taking into account greedy selection and the probability of making worse decisions
void C_AO_BOA::Revision ()
{
  int bestIND = -1;

  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > fB)
    {
      fB = a [i].f;
      bestIND = i;
    }

    if (a [i].f > a [i].fB)
    {
      a [i].fB = a [i].f;
      ArrayCopy (a [i].cB, a [i].c, 0, 0, WHOLE_ARRAY);
    }
  }

  if (bestIND != -1) ArrayCopy (cB, a [bestIND].c, 0, 0, WHOLE_ARRAY);

  S_AO_Agent aT []; ArrayResize (aT, popSize);
  u.Sorting_fB (a, aT, popSize);
}
//——————————————————————————————————————————————————————————————————————————————


Test results
Now let's see how the BOA algorithm works:
BOA|Billiards Optimization Algorithm|50.0|8.0|
=============================
5 Hilly's; Func runs: 10000; result: 0.63960620766331
25 Hilly's; Func runs: 10000; result: 0.3277725645995603
500 Hilly's; Func runs: 10000; result: 0.2514878043770147
=============================
5 Forest's; Func runs: 10000; result: 0.3885662762060409
25 Forest's; Func runs: 10000; result: 0.1955657530616877
500 Forest's; Func runs: 10000; result: 0.15336230733273673
=============================
5 Megacity's; Func runs: 10000; result: 0.5415384615384615
25 Megacity's; Func runs: 10000; result: 0.19046153846153846
500 Megacity's; Func runs: 10000; result: 0.10530769230769324
=============================
All score: 2.79367 (31.04%)
As you can see, the algorithm's performance is quite weak and it does not make it into our ranking table, so I decided to take a closer look at the algorithm and came up with some ideas on how to make it work. Let's look at the equation for the movement of balls again:

X_new = X + rnd [0.0; 1.0] × (P - I × X)

In this equation, the I ratio is applied to the value of the current coordinate of the ball, which has unclear physical meaning, since in fact the scaling is applied to the absolute value of the coordinate. The natural thing to do is to factor this ratio out to allow scaling to the difference between the pocket and the ball coordinate value. Then the resulting record describes a truly physical meaning, either the ball will not reach the pocket, or it will fly over it. The variability is provided by an additional noise factor of a random number in the range [0.0, 1.0].

As a result, we obtain the equation for the movement of balls:

X_new = X + rnd [0.0; 1.0] × (P -X) × I

So, below is the complete code for the modified version of the Moving () method, which shows the commented out string by author's equation followed by my equation version.

//——————————————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm
void C_AO_BOA::Moving ()
{
  //----------------------------------------------------------------------------
  // Initial initialization
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        a [p].cB [c] = a [p].c [c];  // Save the initial position as the best individual solution 
      }
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  for (int p = 0; p < popSize; p++)
  {
    for (int c = 0; c < coords; c++)
    {
      int pocketID = u.RNDminusOne (numPockets);

      //a [p].c [c] = a [p].cB [c] + u.RNDprobab () * (a [pocketID].cB [c] - u.RNDintInRange (1, 2) * a [p].cB [c]);
      a [p].c [c] = a [p].cB [c] + u.RNDprobab () * (a [pocketID].cB [c] - a [p].cB [c]) * u.RNDintInRange (1, 2);
      a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
Now, after the changes, let's see how the algorithm works with the parameters the authors' version showed the best results with:

BOA|Billiards Optimization Algorithm|50.0|8.0|
=============================
5 Hilly's; Func runs: 10000; result: 0.8727603657603271
25 Hilly's; Func runs: 10000; result: 0.7117647027521633
500 Hilly's; Func runs: 10000; result: 0.25339119302510993
=============================
5 Forest's; Func runs: 10000; result: 0.9228482722678735
25 Forest's; Func runs: 10000; result: 0.7601448268715335
500 Forest's; Func runs: 10000; result: 0.3498925749480034
=============================
5 Megacity's; Func runs: 10000; result: 0.6184615384615385
25 Megacity's; Func runs: 10000; result: 0.45876923076923076
500 Megacity's; Func runs: 10000; result: 0.14586153846153965
=============================
All score: 5.09389 (56.60%)

After conducting a few more experiments, I came up with parameters that produced even better results:

BOA|Billiards Optimization Algorithm|50.0|25.0|
=============================
5 Hilly's; Func runs: 10000; result: 0.957565927297626
25 Hilly's; Func runs: 10000; result: 0.8259872884790693
500 Hilly's; Func runs: 10000; result: 0.2523458952211869
=============================
5 Forest's; Func runs: 10000; result: 0.9999999999999929
25 Forest's; Func runs: 10000; result: 0.900362056289584
500 Forest's; Func runs: 10000; result: 0.305018130407844
=============================
5 Megacity's; Func runs: 10000; result: 0.7353846153846153
25 Megacity's; Func runs: 10000; result: 0.5252307692307692
500 Megacity's; Func runs: 10000; result: 0.09563076923077005
=============================
All score: 5.59753 (62.19%)

Let's look at the visualization of the BOA algorithm running on test functions. No particular characteristic behavior is observed in the search space; the movements of the "balls" appear chaotic. It is particularly striking that the algorithm successfully copes with problems of small and medium dimensions, however, large dimension problems show convergence issues. This is especially noticeable on the smooth Hilly function, where the performance is even worse than on the discrete Megacity, which is an extremely rare phenomenon compared to other population-based algorithms. It is also worth noting the algorithm's tendency to get stuck in local minima when solving small dimension problems.

Hilly

BOA on the Hilly test function

Forest

BOA on the Forest test function

Megacity

BOA on the Megacity test function

Let's summarize the test results and look at the efficiency. The algorithm turned out to be quite efficient taking the 8th place in the ranking of the best optimization algorithms, despite having serious shortcomings.

#	AO	Description	Hilly	Hilly final	Forest	Forest final	Megacity (discrete)	Megacity final	Final result	% of MAX
10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)
1	ANS	across neighbourhood search	0.94948	0.84776	0.43857	2.23581	1.00000	0.92334	0.39988	2.32323	0.70923	0.63477	0.23091	1.57491	6.134	68.15
2	CLA	code lock algorithm (joo)	0.95345	0.87107	0.37590	2.20042	0.98942	0.91709	0.31642	2.22294	0.79692	0.69385	0.19303	1.68380	6.107	67.86
3	AMOm	animal migration ptimization M	0.90358	0.84317	0.46284	2.20959	0.99001	0.92436	0.46598	2.38034	0.56769	0.59132	0.23773	1.39675	5.987	66.52
4	(P+O)ES	(P+O) evolution strategies	0.92256	0.88101	0.40021	2.20379	0.97750	0.87490	0.31945	2.17185	0.67385	0.62985	0.18634	1.49003	5.866	65.17
5	CTA	comet tail algorithm (joo)	0.95346	0.86319	0.27770	2.09435	0.99794	0.85740	0.33949	2.19484	0.88769	0.56431	0.10512	1.55712	5.846	64.96
6	TETA	time evolution travel algorithm (joo)	0.91362	0.82349	0.31990	2.05701	0.97096	0.89532	0.29324	2.15952	0.73462	0.68569	0.16021	1.58052	5.797	64.41
7	SDSm	stochastic diffusion search M	0.93066	0.85445	0.39476	2.17988	0.99983	0.89244	0.19619	2.08846	0.72333	0.61100	0.10670	1.44103	5.709	63.44
8	BOAm	billiards optimization algorithm M	0.95757	0.82599	0.25235	2.03590	1.00000	0.90036	0.30502	2.20538	0.73538	0.52523	0.09563	1.35625	5.598	62.19
9	AAm	archery algorithm M	0.91744	0.70876	0.42160	2.04780	0.92527	0.75802	0.35328	2.03657	0.67385	0.55200	0.23738	1.46323	5.548	61.64
10	ESG	evolution of social groups (joo)	0.99906	0.79654	0.35056	2.14616	1.00000	0.82863	0.13102	1.95965	0.82333	0.55300	0.04725	1.42358	5.529	61.44
11	SIA	simulated isotropic annealing (joo)	0.95784	0.84264	0.41465	2.21513	0.98239	0.79586	0.20507	1.98332	0.68667	0.49300	0.09053	1.27020	5.469	60.76
12	ACS	artificial cooperative search	0.75547	0.74744	0.30407	1.80698	1.00000	0.88861	0.22413	2.11274	0.69077	0.48185	0.13322	1.30583	5.226	58.06
13	DA	dialectical algorithm	0.86183	0.70033	0.33724	1.89940	0.98163	0.72772	0.28718	1.99653	0.70308	0.45292	0.16367	1.31967	5.216	57.95
14	BHAm	black hole algorithm M	0.75236	0.76675	0.34583	1.86493	0.93593	0.80152	0.27177	2.00923	0.65077	0.51646	0.15472	1.32195	5.196	57.73
15	ASO	anarchy society optimization	0.84872	0.74646	0.31465	1.90983	0.96148	0.79150	0.23803	1.99101	0.57077	0.54062	0.16614	1.27752	5.178	57.54
16	RFO	royal flush optimization (joo)	0.83361	0.73742	0.34629	1.91733	0.89424	0.73824	0.24098	1.87346	0.63154	0.50292	0.16421	1.29867	5.089	56.55
17	AOSm	atomic orbital search M	0.80232	0.70449	0.31021	1.81702	0.85660	0.69451	0.21996	1.77107	0.74615	0.52862	0.14358	1.41835	5.006	55.63
18	TSEA	turtle shell evolution algorithm (joo)	0.96798	0.64480	0.29672	1.90949	0.99449	0.61981	0.22708	1.84139	0.69077	0.42646	0.13598	1.25322	5.004	55.60
19	DE	differential evolution	0.95044	0.61674	0.30308	1.87026	0.95317	0.78896	0.16652	1.90865	0.78667	0.36033	0.02953	1.17653	4.955	55.06
20	CRO	chemical reaction optimization	0.94629	0.66112	0.29853	1.90593	0.87906	0.58422	0.21146	1.67473	0.75846	0.42646	0.12686	1.31178	4.892	54.36
21	BIO	blood inheritance optimization (joo)	0.81568	0.65336	0.30877	1.77781	0.89937	0.65319	0.21760	1.77016	0.67846	0.47631	0.13902	1.29378	4.842	53.80
22	BSA	bird swarm algorithm	0.89306	0.64900	0.26250	1.80455	0.92420	0.71121	0.24939	1.88479	0.69385	0.32615	0.10012	1.12012	4.809	53.44
23	HS	harmony search	0.86509	0.68782	0.32527	1.87818	0.99999	0.68002	0.09590	1.77592	0.62000	0.42267	0.05458	1.09725	4.751	52.79
24	SSG	saplings sowing and growing	0.77839	0.64925	0.39543	1.82308	0.85973	0.62467	0.17429	1.65869	0.64667	0.44133	0.10598	1.19398	4.676	51.95
25	BCOm	bacterial chemotaxis optimization M	0.75953	0.62268	0.31483	1.69704	0.89378	0.61339	0.22542	1.73259	0.65385	0.42092	0.14435	1.21912	4.649	51.65
26	ABO	african buffalo optimization	0.83337	0.62247	0.29964	1.75548	0.92170	0.58618	0.19723	1.70511	0.61000	0.43154	0.13225	1.17378	4.634	51.49
27	(PO)ES	(PO) evolution strategies	0.79025	0.62647	0.42935	1.84606	0.87616	0.60943	0.19591	1.68151	0.59000	0.37933	0.11322	1.08255	4.610	51.22
28	TSm	tabu search M	0.87795	0.61431	0.29104	1.78330	0.92885	0.51844	0.19054	1.63783	0.61077	0.38215	0.12157	1.11449	4.536	50.40
29	BSO	brain storm optimization	0.93736	0.57616	0.29688	1.81041	0.93131	0.55866	0.23537	1.72534	0.55231	0.29077	0.11914	0.96222	4.498	49.98
30	WOAm	wale optimization algorithm M	0.84521	0.56298	0.26263	1.67081	0.93100	0.52278	0.16365	1.61743	0.66308	0.41138	0.11357	1.18803	4.476	49.74
31	AEFA	artificial electric field algorithm	0.87700	0.61753	0.25235	1.74688	0.92729	0.72698	0.18064	1.83490	0.66615	0.11631	0.09508	0.87754	4.459	49.55
32	AEO	artificial ecosystem-based optimization algorithm	0.91380	0.46713	0.26470	1.64563	0.90223	0.43705	0.21400	1.55327	0.66154	0.30800	0.28563	1.25517	4.454	49.49
33	ACOm	ant colony optimization M	0.88190	0.66127	0.30377	1.84693	0.85873	0.58680	0.15051	1.59604	0.59667	0.37333	0.02472	0.99472	4.438	49.31
34	BFO-GA	bacterial foraging optimization - ga	0.89150	0.55111	0.31529	1.75790	0.96982	0.39612	0.06305	1.42899	0.72667	0.27500	0.03525	1.03692	4.224	46.93
35	SOA	simple optimization algorithm	0.91520	0.46976	0.27089	1.65585	0.89675	0.37401	0.16984	1.44060	0.69538	0.28031	0.10852	1.08422	4.181	46.45
36	ABHA	artificial bee hive algorithm	0.84131	0.54227	0.26304	1.64663	0.87858	0.47779	0.17181	1.52818	0.50923	0.33877	0.10397	0.95197	4.127	45.85
37	ACMO	atmospheric cloud model optimization	0.90321	0.48546	0.30403	1.69270	0.80268	0.37857	0.19178	1.37303	0.62308	0.24400	0.10795	0.97503	4.041	44.90
38	ADAMm	adaptive moment estimation M	0.88635	0.44766	0.26613	1.60014	0.84497	0.38493	0.16889	1.39880	0.66154	0.27046	0.10594	1.03794	4.037	44.85
39	CGO	chaos game optimization	0.57256	0.37158	0.32018	1.26432	0.61176	0.61931	0.62161	1.85267	0.37538	0.21923	0.19028	0.78490	3.902	43.35
40	ATAm	artificial tribe algorithm M	0.71771	0.55304	0.25235	1.52310	0.82491	0.55904	0.20473	1.58867	0.44000	0.18615	0.09411	0.72026	3.832	42.58
41	ASHA	artificial showering algorithm	0.89686	0.40433	0.25617	1.55737	0.80360	0.35526	0.19160	1.35046	0.47692	0.18123	0.09774	0.75589	3.664	40.71
42	ASBO	adaptive social behavior optimization	0.76331	0.49253	0.32619	1.58202	0.79546	0.40035	0.26097	1.45677	0.26462	0.17169	0.18200	0.61831	3.657	40.63
43	MEC	mind evolutionary computation	0.69533	0.53376	0.32661	1.55569	0.72464	0.33036	0.07198	1.12698	0.52500	0.22000	0.04198	0.78698	3.470	38.55
44	CSA	circle search algorithm	0.66560	0.45317	0.29126	1.41003	0.68797	0.41397	0.20525	1.30719	0.37538	0.23631	0.10646	0.71815	3.435	38.17
45	IWO	invasive weed optimization	0.72679	0.52256	0.33123	1.58058	0.70756	0.33955	0.07484	1.12196	0.42333	0.23067	0.04617	0.70017	3.403	37.81
RW	random walk	0.48754	0.32159	0.25781	1.06694	0.37554	0.21944	0.15877	0.75375	0.27969	0.14917	0.09847	0.52734	2.348	26.09


Summary
The modified billiard optimization algorithm (BOAm) showed interesting results on test functions. Analysis of the presented data shows that the algorithm achieves the best results on small and medium-sized problems, gaining maximum values in the Hilly (0.957), Forest (0.999) and Megacity (0.735) tests when reaching 10,000 iterations. This demonstrates its high efficiency in finding optimal solutions for problems of moderate complexity. However, performance drops significantly as the problem size increases, as seen in the results for the 1000-variable scenarios, where the scores drop to 0.252, 0.305, and 0.095, respectively.

It is especially important to note the significant performance improvement in the modified version of the algorithm, which achieves 62.19% of the maximum possible result, which is double the original version's 31.04%. This impressive improvement was achieved by changing just one line of code, which concerns the equation for updating the balls' positions.

The simplicity of the algorithm is both its advantage and its limitation — it is intuitive, easy to implement, and based on an elegant billiards concept — but may require additional modifications to handle high-dimensional problems efficiently. Overall, ranking in the top ten algorithms of the ranking table, BOAm represents a promising metaheuristic approach with a good balance between exploration and exploitation of the solution space.

Table

Figure 2. Color gradation of algorithms according to the corresponding tests

Chart

Figure 3. Histogram of algorithm testing results (scale from 0 to 100, the higher the better, where 100 is the maximum possible theoretical result, in the archive there is a script for calculating the rating table)

BOA pros and cons:

Pros:

Very few external parameters.
Simple implementation.
Performs well on small and medium-sized problems.
Excellent results on problems with "sharp" extremes (such as the Forest function).
Cons:

Gets stuck at local extremes on low-dimensional problems.
Very low speed and accuracy of convergence on "smooth" problems (such as the Hilly function) of high dimension.
The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

github: https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5

Programs used in the article
#	Name	Type	Description
1	#C_AO.mqh
Include
Parent class of population optimization
algorithms
2	#C_AO_enum.mqh
Include
Enumeration of population optimization algorithms
3	TestFunctions.mqh
Include
Library of test functions
4
TestStandFunctions.mqh
Include
Test stand function library
5
Utilities.mqh
Include
Library of auxiliary functions
6
CalculationTestResults.mqh
Include
Script for calculating results in the comparison table
7
Testing AOs.mq5
Script	The unified test stand for all population optimization algorithms
8
Simple use of population optimization algorithms.mq5
Script
A simple example of using population optimization algorithms without visualization
9
Test_AO_BOAm.mq5
Script	BOAm test stand
Translated from Russian by MetaQuotes Ltd.



REAS 11
Successful Restaurateur Algorithm (SRA)
MetaTrader 5 — Tester | 12 January 2026 at 14:54
2 185
0
Andrey Dik
Andrey Dik
Contents
Introduction
Implementation of the algorithm
Test results


Introduction

I have always been fascinated by the parallels between optimization problems and real-life scenarios. While exploring new approaches to metaheuristic algorithms, I discovered similarities between population optimization and the evolution of the restaurant business, and this idea became the inspiration for what I call the Successful Restaurateur Algorithm (SRA).

Imagine a restaurant owner who is constantly striving to improve the menu to increase the restaurant's popularity and attract new customers. Instead of completely eliminating unpopular dishes, our restaurant owner takes a more subtle approach - identifying the least popular item and then carefully mixing it with elements from the most successful dishes. Sometimes conservative changes are made, and sometimes bold new ingredients are added. The goal is always the same: to turn the weakest offering into something that can become a new menu favorite for restaurant customers.

This culinary metaphor forms the basis of SRA. Unlike traditional evolutionary algorithms, which often completely discard poor performers, SRA focuses on rehabilitating poor performers by pairing them with successful elements. This approach preserves diversity in the solution space while steadily improving the overall quality of the population.

In this article, I will cover the basic mechanics of SRA, analyze its implementation, and how parameters like "temperature" and "intensity of cooking experiments" control the balance between exploitation and exploration. I will also share benchmark results comparing SRA to other well-known algorithms in the league table on various test functions.

What began as a creative thought experiment has evolved into a promising approach with unique characteristics. I invite you to explore how this restaurant-inspired algorithm offers optimization solutions with its own unique flavor.



Implementation of the algorithm
Let's understand how the algorithm works through simple analogies. Imagine that I am the owner of a restaurant. I have a menu with different dishes and I notice that some of them are very popular, while some dishes are almost never ordered by guests. What am I going to do? I do not immediately remove unpopular dishes from the menu (thereby reducing the list of available dishes), but instead I take the least popular dish and try to improve it. How? I look at the restaurant's hit items and borrow some ideas or ingredients from there. For example, my fish dish does not sell well, but my salad is very popular. I take elements from a successful salad (maybe a special dressing or serving method) and apply them to the fish dish. It turns out to be something new.

Sometimes I make small changes, and sometimes I decide to try bold experiments. In the beginning, when I first opened the restaurant, I experimented more, but when I found a few really successful dishes, I began to fine-tune the composition of ingredients and their quantities. Over time, even the weakest dishes on my menu get better and better. And sometimes it happens that a former outsider, after some tweaking, becomes a new successful dish! Thus, the overall popularity of the restaurant grows due to the fact that all menu items are successful.

This is how my algorithm works: I do not throw out bad solutions, but constantly improve them, using ideas from the best solutions. And the further we go, the less we experiment, the more we refine what we have already found. Figure 1 shows the algorithm operation diagram.

sra-algorithm-flow

Figure 1. SRA algorithm operation diagram

The diagram starts with the Initialization block, where the initial menu is created. This is followed by the main loop of the algorithm, which centers on the restaurant current menu, sorted by dish quality. This menu is depicted as a color gradient from green (best dishes) to red (worst dishes). Below are four sequential steps: first, selecting dishes for improvement, taking the worst dish and the best donor with increased probability according to a quadratic law; second, creating new variants by combining recipes and mutating ingredients (the higher the temperature, the more daring the experiments); third, evaluating new dishes by calculating a fitness function; fourth, lowering the temperature to reduce the radicality of the experiments. On the left, the dotted arrow shows that the process is repeated until convergence is achieved or the stopping criterion is met. On the right are the designations: A (green circle) - the best dishes, B (red circle) - the worst dishes. The entire diagram visualizes a process reminiscent of a restaurateur who systematically improves weaker items on the menu by using elements of successful dishes.

Let's move on to writing the pseudocode for the SRA algorithm.

// Initialization
Create a population of popSize agents (menu items)
For each agent:
    Randomly initialize all coordinates within the allowed range
Set initial temperature = 1.0
Set cooling ratio = 0.98
Set the intensity of culinary experiments = 0.3

// The main loop of the algorithm
UNTIL the stopping condition is met:
    // Step 1: Rating of all agents
    For each agent:
        Calculate the value of the fitness function
    
    // Merge the current and previous populations
    Create a common menu from current agents and previous agents
    Sort the general menu by fitness function value from best to worst
    
    // Step 2: Creating new options
    For each agent in the new population:
        // Take the worst element from the first half of the sorted population
        Copy coordinates from the agent with index (popSize-1)
        
        // Choose between improvement and experimentation
        If (random number < (1.0 - menuInnovationRate * temperature)):
            // We select a "donor" using the quadratic roulette method
            r = random number from 0 to 1
            r = r²
            donorIndex = scale r from 0 to (popSize-1)
            
            // For each coordinate
            For each c coordinate:
                // We take the coordinate from the donor with the probability of 0.8
                If (random number < 0.8):
                    current_agent.c = donor.c
                
                // Adaptive mutation
                mutationRate = 0.1 + 0.4 * temperature * (agent_index / popSize)
                If (random number < mutationRate):
                    // Select the mutation type
                    If (random number < 0.5):
                        current_agent.c = gaussian distribution(current_agent.c)
                    Otherwise:
                        current_agent.c = random value in range
                    
                    // Make sure the value is within acceptable limits
                    current_agent.c = round to nearest valid value
        Otherwise:
            // Creating a new "dish"
            For each c coordinate:
                If (random number < 0.7):
                    current_agent.c = random value in range
                Otherwise:
                    current_agent.c = gaussian distribution(best_solution.c)
                
                // Make sure the value is within acceptable limits
                current_agent.c = round to the nearest valid value
        
        // Elitism - an occasional simple addition of elements from a better solution
        If (random number < 0.1):
            numEliteCoords = random number from 1 to (coords * 0.3)
            For i from 1 to numEliteCoords:
                c = random coordinate index
                current_agent.c = best_solution.c
    
    // Step 3: Update the best solution
    For each agent:
        If agent.fitness > best_solution.fitness:
            best_solution = agent
    
    // Step 4: Temperature drop
    temperature = temperature * cooling_ratio
    If temperature < 0.1:
        temperature = 0.1

Return best_solution

Now we can start writing the algorithm code. Write the C_AO_SRA class that inherits from the C_AO main class and implements the SRA algorithm. Let's take a closer look at it.

Constructor and destructor: popSize, temperature, coolingRate, menuInnovationRate — these parameters determine the main characteristics of the algorithm, such as the number of agents and search control parameters.

SetParams method: updates the class parameters based on the values stored in the "params" array, the parameters previously initialized in the constructor.

Init method: intended to initialize the algorithm. It takes the minimum and maximum values, the step size of the parameters and the number of epochs and prepares the algorithm to perform the search task.

Moving and Revision methods: designed to perform the main stages of the algorithm operation related to the movement (or updating) of the state of agents and "Revision", which is responsible for revising and adapting the parameters.

Class members:

temperature — current temperature associated with the study control and the temperature chart of the algorithm.
coolingRate — cooling rate used to control how quickly the temperature will drop.
menuInnovationRate — intensity of culinary experiments, the extent to which agents are encouraged to search for new solutions.
Private class members:
S_AO_Agent menu [] — array of agents that represents a "menu" in the context of the SRA algorithm.
S_AO_Agent menuT [] — array of agents used to temporarily store dish options.
//——————————————————————————————————————————————————————————————————————————————
class C_AO_SRA : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_SRA () { }
  C_AO_SRA ()
  {
    ao_name = "SRA";
    ao_desc = "Successful Restaurateur Algorithm (joo)";
    ao_link = "https://www.mql5.com/en/articles/17380";

    popSize            = 50;   // number of agents (size of the "menu")
    temperature        = 1.0;  // initial "temperature" for research control
    coolingRate        = 0.98; // cooling rate
    menuInnovationRate = 0.3;  // intensity of culinary experiments

    ArrayResize (params, 4);

    params [0].name = "popSize";            params [0].val = popSize;
    params [1].name = "temperature";        params [1].val = temperature;
    params [2].name = "coolingRate";        params [2].val = coolingRate;
    params [3].name = "menuInnovationRate"; params [3].val = menuInnovationRate;
  }

  void SetParams ()
  {
    popSize            = (int)params [0].val;
    temperature        = params      [1].val;
    coolingRate        = params      [2].val;
    menuInnovationRate = params      [3].val;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  double temperature;        // current "temperature"
  double coolingRate;        // cooling rate 
  double menuInnovationRate; // intensity of culinary experiments

  private: //-------------------------------------------------------------------
  S_AO_Agent menu  [];
  S_AO_Agent menuT [];
};
//——————————————————————————————————————————————————————————————————————————————

The Init method of the C_AO_SRA class initializes the algorithm:

Check initialization: The method calls StandardInit with the minimum and maximum values of the ranges and steps, if unsuccessful, "false" is returned.

Initializing arrays:

Distributes the sizes of the "menu" and "menuT" arrays according to the number of agents.
Initializes each agent in the "menu" array.
Temperature reset: sets the initial value of "temperature" to "1.0".

Successful completion: returns "true" if initialization was successful.

//——————————————————————————————————————————————————————————————————————————————
//--- Initialization
bool C_AO_SRA::Init (const double &rangeMinP  [],
                     const double &rangeMaxP  [],
                     const double &rangeStepP [],
                     const int epochsP = 0)
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  ArrayResize (menu,  popSize * 2);
  ArrayResize (menuT, popSize * 2);

  for (int p = 0; p < popSize * 2; p++) menu [p].Init (coords);

  temperature = 1.0; // reset temperature during initialization

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
The Moving method of the C_AO_SRA class implements the main step of the algorithm. It has two main parts: initialization of agents and their adaptation through mutation and creation of new solutions.

Initial initialization (if "revision" is "false"):
Each agent is initialized with random values within given ranges (rangeMin, rangeMax) using steps (rangeStep). The values are stored in "c" and "cB" for each agent.
"revision" is set to "true" and the method terminates.
Temperature drop: the temperature is multiplied by the cooling ratio (coolingRate), which affects the likelihood of further changes.

The main loop by agents: for each agent, the worst element from the first half of the sorted population (from the menu array) is selected.

Classification of actions: with a certain probability, which depends on the temperature, the agent either:

modifies the current solution (using the "donor recipe" from the best dish), applying a mutation with varying probability.
creates a new solution (random values).
Elitism: with some probability, elements from the best solution found can be added to the new solution.

//——————————————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm
void C_AO_SRA::Moving ()
{
  //----------------------------------------------------------------------------
  // Initial initialization
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        a [p].cB [c] = a [p].c [c];
      }
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  // Lower the temperature
  temperature *= coolingRate;

  // Main loop on population agents
  for (int p = 0; p < popSize; p++)
  {
    // Take the worst element from the first half of the sorted population (with index popSize-1)
    // Remember that items are sorted from best to worst in the menu
    ArrayCopy (a [p].c, menu [popSize - 1].c, 0, 0, WHOLE_ARRAY);

    // Decide whether to create a hybrid or experiment with a new "dish"
    // The probability of an experiment depends on the temperature - there are more experiments at the beginning
    if (u.RNDprobab () < (1.0 - menuInnovationRate * temperature))
    {
      // Select a "donor-recipe" with a probability proportional to the success of the dish
      double r = u.RNDprobab ();
      r = pow (r, 2);                                         // Increased preference for better dishes
      int menuIND = (int)u.Scale (r, 0, 1.0, 0, popSize - 1); // The best ones are at the beginning of the array 

      // For each coordinate
      for (int c = 0; c < coords; c++)
      {
        // Take the parameter from a successful dish with the probability depending on the temperature
        if (u.RNDprobab () < 0.8)
        {
          a [p].c [c] = menu [menuIND].c [c];
        }

        // Mutation with adaptive probability - the further from the best solution and the higher the temperature, the more mutations
        double mutationRate = 0.1 + 0.4 * temperature * (double)(p) / popSize;
        if (u.RNDprobab () < mutationRate)
        {
          // Combination of different types of mutations
          if (u.RNDprobab () < 0.5) a [p].c [c] = u.GaussDistribution (a [p].c [c], rangeMin [c], rangeMax [c], 2);
          else                      a [p].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]); // Sometimes a completely new value

          // Make sure the value is within acceptable limits
          a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        }
      }
    }
    else // Create a completely new "dish"
    {
      for (int c = 0; c < coords; c++)
      {
        // Variation 1: Completely random value
        if (u.RNDprobab () < 0.7)
        {
          a [p].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        }
        // Variation 2: based on the best solution found with a large deviation
        else
        {
          a [p].c [c] = u.GaussDistribution (cB [c], rangeMin [c], rangeMax [c], 1);
        }

        a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      }
    }

    // Sometimes we add elements from the best solution directly (elitism)
    if (u.RNDprobab () < 0.1)
    {
      int numEliteCoords = u.RNDintInRange (1, coords / 3); // Take from 1 to 30% of the coordinates
      for (int i = 0; i < numEliteCoords; i++)
      {
        int c = u.RNDminusOne (coords);
        a [p].c [c] = cB [c]; // Take the value from the best solution
      }
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
The Revision method of the C_AO_SRA class is responsible for updating the best solution found and for managing the overall "menu" of solutions during the algorithm operation:

Finding the best agent: iterates through all agents in the current population and finds the agent with the best fitness function (f). If a new best agent is found, update the fB value and the bestIND index.

Updating the best solution: if the best agent is found (i.e. bestIND is not equal to -1), its decision parameters (c) are copied into the cB variable representing the current best decision.

Updating the general "menu": adds the current parameters of all agents to the general "menu", which allows you to save completed experiments.

Sorting the menu: sorts the "menu" array by fitness function from best to worst solutions, ensuring that the best solutions are in the first half. This will be used in the next iteration of the algorithm.

Temperature control: sets a lower threshold for the temperature so that it does not fall below "0.1", which prevents it from converging too quickly.
//——————————————————————————————————————————————————————————————————————————————
//--- Update the best solution taking into account greedy selection and the probability of making worse decisions
void C_AO_SRA::Revision ()
{
  int bestIND = -1;

  // Find the best agent in the current population
  for (int p = 0; p < popSize; p++)
  {
    if (a [p].f > fB)
    {
      fB = a [p].f;
      bestIND = p;
    }
  }

  // If we find a better solution, update cB
  if (bestIND != -1) ArrayCopy (cB, a [bestIND].c, 0, 0, WHOLE_ARRAY);

  // Add the current set of dishes to the general "menu"
  for (int p = 0; p < popSize; p++)
  {
    menu [popSize + p] = a [p];
  }

  // Sort the entire "menu" from best to worst solutions
  // After sorting, the first half of the menu will contain the best solutions,
  // which will be used in the next iteration
  u.Sorting (menu, menuT, popSize * 2);

  // Prevent the temperature from falling below a certain threshold
  if (temperature < 0.1) temperature = 0.1;
}
//——————————————————————————————————————————————————————————————————————————————


Test results
Now we can see how the SRA algorithm works. Below are the test results:

SRA|Successful Restaurateur Algorithm|50.0|1.0|0.98|0.3|
=============================
5 Hilly's; Func runs: 10000; result: 0.9688326305968623
25 Hilly's; Func runs: 10000; result: 0.6345483084017249
500 Hilly's; Func runs: 10000; result: 0.292167027537253
=============================
5 Forest's; Func runs: 10000; result: 0.946368863880973
25 Forest's; Func runs: 10000; result: 0.5550607959254661
500 Forest's; Func runs: 10000; result: 0.19124225531141872
=============================
5 Megacity's; Func runs: 10000; result: 0.7492307692307693
25 Megacity's; Func runs: 10000; result: 0.4403076923076923
500 Megacity's; Func runs: 10000; result: 0.12526153846153956
=============================
All score: 4.90302 (54.48%)

Visualization of the SRA algorithm operation on the test stand allows us to draw conclusions about the characteristic features of the search strategy. In this case, we observe a broad exploration of the search space: agents are evenly distributed throughout the space, exploring even its most remote corners. At the same time, there are no noticeable signs of grouping at local extremes; the movements of agents appear chaotic.

However, despite its good exploration ability, the algorithm exhibits some shortcomings in refining solutions, which is reflected in the relatively low convergence accuracy. It should also be noted that there is a small spread in the test results.

Hilly

SRA on the Hilly test function

Forest

SRA on the Forest test function

Megacity

SRA on the Megacity test function

Based on the test results, the algorithm ranks 20th in the ranking of the strongest population optimization algorithms. Currently, the table presents nine proprietary optimization algorithms (joo), including the new SRA algorithm.

#	AO	Description	Hilly	Hilly final	Forest	Forest final	Megacity (discrete)	Megacity final	Final result	% of MAX
10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)
1	ANS	across neighbourhood search	0.94948	0.84776	0.43857	2.23581	1.00000	0.92334	0.39988	2.32323	0.70923	0.63477	0.23091	1.57491	6.134	68.15
2	CLA	code lock algorithm (joo)	0.95345	0.87107	0.37590	2.20042	0.98942	0.91709	0.31642	2.22294	0.79692	0.69385	0.19303	1.68380	6.107	67.86
3	AMOm	animal migration ptimization M	0.90358	0.84317	0.46284	2.20959	0.99001	0.92436	0.46598	2.38034	0.56769	0.59132	0.23773	1.39675	5.987	66.52
4	(P+O)ES	(P+O) evolution strategies	0.92256	0.88101	0.40021	2.20379	0.97750	0.87490	0.31945	2.17185	0.67385	0.62985	0.18634	1.49003	5.866	65.17
5	CTA	comet tail algorithm (joo)	0.95346	0.86319	0.27770	2.09435	0.99794	0.85740	0.33949	2.19484	0.88769	0.56431	0.10512	1.55712	5.846	64.96
6	TETA	time evolution travel algorithm (joo)	0.91362	0.82349	0.31990	2.05701	0.97096	0.89532	0.29324	2.15952	0.73462	0.68569	0.16021	1.58052	5.797	64.41
7	SDSm	stochastic diffusion search M	0.93066	0.85445	0.39476	2.17988	0.99983	0.89244	0.19619	2.08846	0.72333	0.61100	0.10670	1.44103	5.709	63.44
8	BOAm	billiards optimization algorithm M	0.95757	0.82599	0.25235	2.03590	1.00000	0.90036	0.30502	2.20538	0.73538	0.52523	0.09563	1.35625	5.598	62.19
9	AAm	archery algorithm M	0.91744	0.70876	0.42160	2.04780	0.92527	0.75802	0.35328	2.03657	0.67385	0.55200	0.23738	1.46323	5.548	61.64
10	ESG	evolution of social groups (joo)	0.99906	0.79654	0.35056	2.14616	1.00000	0.82863	0.13102	1.95965	0.82333	0.55300	0.04725	1.42358	5.529	61.44
11	SIA	simulated isotropic annealing (joo)	0.95784	0.84264	0.41465	2.21513	0.98239	0.79586	0.20507	1.98332	0.68667	0.49300	0.09053	1.27020	5.469	60.76
12	ACS	artificial cooperative search	0.75547	0.74744	0.30407	1.80698	1.00000	0.88861	0.22413	2.11274	0.69077	0.48185	0.13322	1.30583	5.226	58.06
13	DA	dialectical algorithm	0.86183	0.70033	0.33724	1.89940	0.98163	0.72772	0.28718	1.99653	0.70308	0.45292	0.16367	1.31967	5.216	57.95
14	BHAm	black hole algorithm M	0.75236	0.76675	0.34583	1.86493	0.93593	0.80152	0.27177	2.00923	0.65077	0.51646	0.15472	1.32195	5.196	57.73
15	ASO	anarchy society optimization	0.84872	0.74646	0.31465	1.90983	0.96148	0.79150	0.23803	1.99101	0.57077	0.54062	0.16614	1.27752	5.178	57.54
16	RFO	royal flush optimization (joo)	0.83361	0.73742	0.34629	1.91733	0.89424	0.73824	0.24098	1.87346	0.63154	0.50292	0.16421	1.29867	5.089	56.55
17	AOSm	atomic orbital search M	0.80232	0.70449	0.31021	1.81702	0.85660	0.69451	0.21996	1.77107	0.74615	0.52862	0.14358	1.41835	5.006	55.63
18	TSEA	turtle shell evolution algorithm (joo)	0.96798	0.64480	0.29672	1.90949	0.99449	0.61981	0.22708	1.84139	0.69077	0.42646	0.13598	1.25322	5.004	55.60
19	DE	differential evolution	0.95044	0.61674	0.30308	1.87026	0.95317	0.78896	0.16652	1.90865	0.78667	0.36033	0.02953	1.17653	4.955	55.06
20	SRA	successful restaurateur algorithm (joo)	0.96883	0.63455	0.29217	1.89555	0.94637	0.55506	0.19124	1.69267	0.74923	0.44031	0.12526	1.31480	4.903	54.48
21	CRO	chemical reaction optimization	0.94629	0.66112	0.29853	1.90593	0.87906	0.58422	0.21146	1.67473	0.75846	0.42646	0.12686	1.31178	4.892	54.36
22	BIO	blood inheritance optimization (joo)	0.81568	0.65336	0.30877	1.77781	0.89937	0.65319	0.21760	1.77016	0.67846	0.47631	0.13902	1.29378	4.842	53.80
23	BSA	bird swarm algorithm	0.89306	0.64900	0.26250	1.80455	0.92420	0.71121	0.24939	1.88479	0.69385	0.32615	0.10012	1.12012	4.809	53.44
24	HS	harmony search	0.86509	0.68782	0.32527	1.87818	0.99999	0.68002	0.09590	1.77592	0.62000	0.42267	0.05458	1.09725	4.751	52.79
25	SSG	saplings sowing and growing	0.77839	0.64925	0.39543	1.82308	0.85973	0.62467	0.17429	1.65869	0.64667	0.44133	0.10598	1.19398	4.676	51.95
26	BCOm	bacterial chemotaxis optimization M	0.75953	0.62268	0.31483	1.69704	0.89378	0.61339	0.22542	1.73259	0.65385	0.42092	0.14435	1.21912	4.649	51.65
27	ABO	african buffalo optimization	0.83337	0.62247	0.29964	1.75548	0.92170	0.58618	0.19723	1.70511	0.61000	0.43154	0.13225	1.17378	4.634	51.49
28	(PO)ES	(PO) evolution strategies	0.79025	0.62647	0.42935	1.84606	0.87616	0.60943	0.19591	1.68151	0.59000	0.37933	0.11322	1.08255	4.610	51.22
29	TSm	tabu search M	0.87795	0.61431	0.29104	1.78330	0.92885	0.51844	0.19054	1.63783	0.61077	0.38215	0.12157	1.11449	4.536	50.40
30	BSO	brain storm optimization	0.93736	0.57616	0.29688	1.81041	0.93131	0.55866	0.23537	1.72534	0.55231	0.29077	0.11914	0.96222	4.498	49.98
31	WOAm	wale optimization algorithm M	0.84521	0.56298	0.26263	1.67081	0.93100	0.52278	0.16365	1.61743	0.66308	0.41138	0.11357	1.18803	4.476	49.74
32	AEFA	artificial electric field algorithm	0.87700	0.61753	0.25235	1.74688	0.92729	0.72698	0.18064	1.83490	0.66615	0.11631	0.09508	0.87754	4.459	49.55
33	AEO	artificial ecosystem-based optimization algorithm	0.91380	0.46713	0.26470	1.64563	0.90223	0.43705	0.21400	1.55327	0.66154	0.30800	0.28563	1.25517	4.454	49.49
34	ACOm	ant colony optimization M	0.88190	0.66127	0.30377	1.84693	0.85873	0.58680	0.15051	1.59604	0.59667	0.37333	0.02472	0.99472	4.438	49.31
35	BFO-GA	bacterial foraging optimization - ga	0.89150	0.55111	0.31529	1.75790	0.96982	0.39612	0.06305	1.42899	0.72667	0.27500	0.03525	1.03692	4.224	46.93
36	SOA	simple optimization algorithm	0.91520	0.46976	0.27089	1.65585	0.89675	0.37401	0.16984	1.44060	0.69538	0.28031	0.10852	1.08422	4.181	46.45
37	ABHA	artificial bee hive algorithm	0.84131	0.54227	0.26304	1.64663	0.87858	0.47779	0.17181	1.52818	0.50923	0.33877	0.10397	0.95197	4.127	45.85
38	ACMO	atmospheric cloud model optimization	0.90321	0.48546	0.30403	1.69270	0.80268	0.37857	0.19178	1.37303	0.62308	0.24400	0.10795	0.97503	4.041	44.90
39	ADAMm	adaptive moment estimation M	0.88635	0.44766	0.26613	1.60014	0.84497	0.38493	0.16889	1.39880	0.66154	0.27046	0.10594	1.03794	4.037	44.85
40	CGO	chaos game optimization	0.57256	0.37158	0.32018	1.26432	0.61176	0.61931	0.62161	1.85267	0.37538	0.21923	0.19028	0.78490	3.902	43.35
41	ATAm	artificial tribe algorithm M	0.71771	0.55304	0.25235	1.52310	0.82491	0.55904	0.20473	1.58867	0.44000	0.18615	0.09411	0.72026	3.832	42.58
42	ASHA	artificial showering algorithm	0.89686	0.40433	0.25617	1.55737	0.80360	0.35526	0.19160	1.35046	0.47692	0.18123	0.09774	0.75589	3.664	40.71
43	ASBO	adaptive social behavior optimization	0.76331	0.49253	0.32619	1.58202	0.79546	0.40035	0.26097	1.45677	0.26462	0.17169	0.18200	0.61831	3.657	40.63
44	MEC	mind evolutionary computation	0.69533	0.53376	0.32661	1.55569	0.72464	0.33036	0.07198	1.12698	0.52500	0.22000	0.04198	0.78698	3.470	38.55
45	CSA	circle search algorithm	0.66560	0.45317	0.29126	1.41003	0.68797	0.41397	0.20525	1.30719	0.37538	0.23631	0.10646	0.71815	3.435	38.17
RW	random walk	0.48754	0.32159	0.25781	1.06694	0.37554	0.21944	0.15877	0.75375	0.27969	0.14917	0.09847	0.52734	2.348	26.09


Summary
Having developed and tested the Successful Restaurateur Algorithm (SRA), I can confidently say that it has proven itself to be a success. Currently, the algorithm ranks 20th in the ranking table, which is quite good for a new concept. While analyzing the results, I noticed some peculiarities in its behavior. For small-dimensional problems, there is a scatter of results. This is especially noticeable in the discrete Megacity function, where the algorithm has a particularly large spread of values. This function is very difficult for algorithms and getting stuck in local extremes is more the rule than the exception.

On high-dimensional problems, SRA shows results that are slightly weaker than expected. This is probably due to the fact that in high-dimensional spaces, the strategy of improving the worst solutions requires more fine-tuning of the temperature and cooling rate parameters.

However, I consider SRA to be a decent algorithm with potential for further improvement. Its culinary metaphor not only makes the algorithm understandable, but also opens the way for intuitive modifications, refinement of the adaptive mutation mechanism, and the possibility of experimenting with various schemes for selecting "donor dishes."

While creating the successful restaurateur algorithm, I sought not so much to achieve superiority over existing optimization methods, but to reveal new conceptual horizons through an original real life metaphor. The results of the study convincingly demonstrate that this approach deserves its place in the ecosystem of metaheuristic algorithms.

The idea of focusing on the worst solution in a population and using it as a basis for experiments — the concept that might seem crazy at first glance — has proven unexpectedly productive. It is precisely this principle of "outsider rehabilitation" that reveals amazing potential for optimization. Like a skilled restaurateur transforming an unpopular dish into a future hit, the algorithm transforms weak solutions into superior ones, using the ingredients of the leaders.

This experience confirms a valuable rule in scientific research: even the most unorthodox ideas, if properly implemented, can bring practical benefits. Unconventional approaches often reveal aspects of a problem that remain unnoticed when using traditional methodologies.

Tab

Figure 2. Color gradation of algorithms according to the corresponding tests

Chart

Figure 3. Histogram of algorithm testing results (scale from 0 to 100, the higher the better, where 100 is the maximum possible theoretical result, in the archive there is a script for calculating the rating table)

SRA pros and cons:

Pros:

Simple implementation.
The results are good.
Cons:

No serious downsides.
The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

github: https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5



READ 12
Neuroboids Optimization Algorithm (NOA)
MetaTrader 5 — Tester | 14 January 2026 at 15:28
1 844
0
Andrey Dik
Andrey Dik
Contents
Introduction
Implementation of the algorithm
Test results


Introduction

While researching optimization algorithms, I was always drawn to the idea of creating the simplest, yet most efficient solutions possible. Observing how nature solves complex problems through the interaction of simple organisms, I developed a new optimization algorithm - the Neuroboids Optimization Algorithm (NOA).

The algorithm is based on the concept of minimalistic neural agents – neuroboids. Each neuroboid is a simple neural network with two layers of neurons, which is trained using the Adam algorithm. The uniqueness of this approach lies in the fact that, despite the extreme simplicity of individual agents, their collective behavior must effectively explore the solution space of complex optimization problems.

The inspiration for NOA came from the processes of self-organization in natural systems, where simple units, following basic rules, form complex adaptive structures. In this article, I present the theoretical justification of the algorithm, its mathematical model, and the results of an experimental study of its efficiency on standard optimization test functions.



Implementation of the algorithm
Imagine that you are walking in the garden after the rain. Earthworms are everywhere - simple creatures with a primitive nervous system. They do not have the ability to "think" in our sense, but somehow they find their way through difficult terrain, avoid danger, find food and partners. Their tiny brains contain only a few thousand neurons, yet they have existed for millions of years. This is how the idea of neuroboids was born.

What if we combined the simplicity of a worm with the power of collective intelligence? In nature, simple organisms achieve incredible results when they work together — ants build complex colonies, bees solve optimization problems when collecting nectar, and flocks of birds form complex dynamic structures without centralized control.

My neuroboids are like these earthworms. Each one has its own small neural network - not some massive architecture with millions of parameters, but just a few neurons at the input and output. They do not know the entire search space, they only see their local environment. When one worm finds a fertile patch of soil rich in nutrients, others gradually gravitate towards that spot. But they do not just follow blindly – each one maintains their own individuality, their own movement strategy. The neuroboids do not need to know all the math behind optimization. They learn on their own, through trial and error. When one of them finds a good solution, the others do not just copy its coordinates, but learn to understand why this solution is good and how to get there on their own.

Remember sunsets at sea? The sun is reflected in millions of glares on the waves, and each glare is its own little story of the interaction of light with water. The same is true for my neuroboids - each one reflects a part of the solution, and together they create a complete picture. From a distance, their movement may appear chaotic. But within this apparent chaos, order is born – a self-organizing system for finding optimal solutions.

The neuroboids also have no central commander. Instead, each uses its own small neural network to decide whether to follow the best known solution or explore new territory, copy a neighbor's successful strategy or risk creating its own. 

In the world obsessed with complexity and scale, neuroboids are a reminder that the greatest systems are often built from the simplest elements. Just as grains of sand form beaches and mountains, and droplets form oceans, so too do my little digital worms, each with its own tiny neural network, work together to solve problems that giant, monolithic algorithms fail to solve.

noa-algorithm-diagram

Figure 1. NOA algorithm operation

Figure 1 shows the main components and operating principle of the NOA algorithm. The search space is the domain in which neuroboids (blue circles) search for the optimal solution. Neuroboids are optimization agents, each with its own neural network. The best solution is the current best solution found (the golden circle) toward which the neuroboids strive. The neural network architecture is as follows: each neuroboid uses its own neural network to determine the direction of movement. The algorithm cyclical process consists of the following stages: initialization — random placement of neuroboids; neural network training — learning based on the best solution found; neuroboids move under the control of the trained neural networks; the best solution is updated if a better one is found. The dotted lines show the directions of movement of the neuroboids, which are determined by the outputs of their neural networks and striving for the best solution with an element of randomness.

The term "neuroboid" combines the concepts of a neural network and a boid (an artificial "bird" in models of collective behavior). Each neuroboid agent is a combination of a position in the solution space
and a personal neural network that determines the movement strategy.

Unlike traditional metaheuristic algorithms (such as genetic algorithms or swarm methods), where the behavior of agents is determined by fixed rules, in NOA: each neuronode has its own neural network, which is trained during the optimization; neural networks gradually learn to determine the most promising search directions, and training occurs based on the best solution found (target vector). It turns out that the optimization algorithm includes two search strategies: the first is NOA itself, and the second is ADAM built into the neural network. The backpropagation mechanism allows ADAM to be used in the context of the NOA algorithm as an independent tool for adjusting the neural network weights of each neuron. Consequently, the total number of weights of all neuron bodies can be much larger than the problem dimension — there is no need to optimize the neural network weights of the entire population; this happens naturally and automatically.

The behavior of neuroboids has certain parallels with: social learning in nature (observation of successful individuals), neuroplasticity (the ability of the nervous system to adapt to changing conditions) and collective intelligence (emergent optimization through the interaction of many simple agents).

The main technical features of the NOA algorithm can be identified:

forward and backward propagation of errors within the optimization,
distributed training of multiple neural networks, each of which forms its own strategy,
adaptive behavior adjustment based on the current search state,
using activation functions of neural networks to create non-linear search dynamics.
This combination makes NOA an interesting hybrid approach that combines machine learning and optimization paradigms. Instead of using neural networks to approximate the objective function (as in surrogate models), NOA uses them to indirectly guide the search itself, creating a kind of "meta-learning" for solving optimization problems.

Now we can write the pseudocode for the NOA algorithm:

Initialization:

Create a population of N neural networks (neuroboids)
Each neural network has a structure with a number of inputs and outputs equal to the dimension of the optimization problem
Set the parameters:
popSize (population size)
actFunc (neuron activation function)
dispScale (displacement scale)
eliteProb (probability of copying elite coordinates)
Algorithm:

If this is the first iteration (revision = false):
For each neuroboid in the population:
Randomly initialize coordinates in the search space
Convert coordinates to acceptable discrete values
Set revision = true and terminate the current iteration
For subsequent iterations:
For most neuroboids (all but the last 5):
For each coordinate:
With the eliteProb probability, replace the coordinate value with the value from the best solution found (cB)
For each neuroboid in the population:
Scale the best solution found (cB) and the current position of the neuroboid to the range [-1, 1]
Perform forward propagation through the neural network of the current neuroboid
Calculate the error between the target value (scaled by cB) and the output of the neural network
Perform backpropagation to train the neural network
Update the coordinates of the neuroboid by shifting them in the direction determined by the output of the neural network
Convert coordinates to acceptable discrete values
Rating and update:
For each neuroboid in the population:
Calculate the value of the objective function for the current coordinates
If the value is better than the best found (fB):
Update best found value (fB)
Save current coordinates as best (cB)
Let's start writing the algorithm code. Define the C_AO_NOA class, which is derived from the C_AO class. This means that it inherits the properties and methods of C_AO and also adds its own. Main elements of the class:

~C_AO_NOA () destructor: removes dynamically allocated activation function objects for each element of the "nn" array - the individual neural network of neuroboids. This helps prevent memory leaks.

C_AO_NOA () constructor:

Initializes the optimization algorithm parameters.
Sets initial values for the parameters: popSize - population size, actFunc - neuron activation function, dispScale - movement scale, probability of copying elite coordinates.
Reserve the "params" array to store parameters.
SetParams () method: sets parameters based on the values stored in the "params" array.

Init () method: defined to initialize the class with the value ranges of rangeMinP, rangeMaxP and rangeStepP, as well as the epochsP number of epochs. 

Moving () and Revision (): these methods serve to move individuals in the population and perform the evaluation and updating of decisions.

nn array: an array of instances of the neural network class of each neuroboid.

Closed methods: ScaleInp () and ScaleOut () - methods for scaling input and output data, respectively.

//——————————————————————————————————————————————————————————————————————————————
class C_AO_NOA : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_NOA ()
  {
    for (int i = 0; i < ArraySize (nn); i++) if (CheckPointer (nn [i].actFunc)) delete nn [i].actFunc;
  }
  C_AO_NOA ()
  {
    ao_name = "NOA";
    ao_desc = "Neuroboids Optimization Algorithm (joo)";
    ao_link = "https://www.mql5.com/en/articles/16992";

    popSize   = 50;   // population size
    actFunc   = 0;    // neuron activation function
    dispScale = 0.01; // scale of movements
    eliteProb = 0.1;  // probability of copying elite coordinates

    ArrayResize (params, 4);
    params [0].name = "popSize";   params [0].val  = popSize;
    params [1].name = "actFunc";   params [1].val  = actFunc;
    params [2].name = "dispScale"; params [2].val  = dispScale;
    params [3].name = "eliteProb"; params [3].val  = eliteProb;
  }

  void SetParams ()
  {
    popSize   = (int)params [0].val;
    actFunc   = (int)params [1].val;
    dispScale = params      [2].val;
    eliteProb = params      [3].val;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  int    actFunc;   // neuron activation function
  double dispScale; // scale of movements
  double eliteProb; // probability of copying elite coordinates

  private: //-------------------------------------------------------------------
  C_MLPa nn [];

  void ScaleInp (double &inp [], double &out []);
  void ScaleOut (double &inp [], double &out []);
};
//——————————————————————————————————————————————————————————————————————————————
The Init method is intended to initialize an instance of the C_AO_NOA class. It takes a number of parameters that specify the range of values and the number of epochs, and performs the necessary operations to tune the algorithm.

Method signature: indicates that the initialization was successful. Parameters:

MinP [] - minimum values for parameters.
MaxP [] - maximum values for parameters.
StepP [] - step for changing parameters.
epochsP - number of epochs.
Standard initialization: the first line of the method calls the StandardInit function with the passed ranges.

Setting up neural network configuration:

The nnConf array is created, which contains the sizes of the input and output layers of the neural network.
The ArrayResize is called to resize the nnConf array to 2 (input and output layer).
Both array elements are initialized with the "coords" value, which corresponds to the dimensions of the input data.
E_Act enumeration declaration: an enumeration is created that defines different neuron activation functions (eActTanh, eActAlgSigm, eActRatSigm etc.).

Backing up the neuron array: resizes the "nn" array (the array of neurons) to popSize, which was previously set in the constructor.

Initialization of neurons:

The cnt variable is initialized from zero and is used to pass a unique random seed to initialize each neuron, ensuring that the weights in each neural network are uniquely initialized.
Each neuron is initialized in the "for" loop. The nn[i].Init() method is called with the nnConf configuration, the actFunc activation function, and a seed based on the current cnt value and the number of milliseconds elapsed since the system started.
The cnt value is incremented with each loop iteration, allowing for unique seeds to be generated for each neuron initialization.
Return the result: if all operations are successful, the method returns "true", the initialization was successful.

The Init method of the С_AO_NOA class is key to setting up the optimization algorithm, providing initialization of the neural network and associated parameters. 

//——————————————————————————————————————————————————————————————————————————————
bool C_AO_NOA::Init (const double &rangeMinP  [], // minimum values
                     const double &rangeMaxP  [], // maximum values
                     const double &rangeStepP [], // step change
                     const int     epochsP = 0)   // number of epochs
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  int nnConf [];
  ArrayResize (nnConf, 2);
  nnConf [0] = coords;
  nnConf [1] = coords;

  enum E_Act
  {
    eActTanh,      //0
    eActAlgSigm,   //1
    eActRatSigm,   //2

    eActSoftPlus,  //3
    eActBentIdent, //4
    eActSiLU,      //5

    eActACON,      //6
    eActSERF,      //7
    eActSnake      //8
  };

  ArrayResize (nn, popSize);
  int cnt = 0;
  for (int i = 0; i < popSize; i++)
  {
    nn [i].Init (nnConf, actFunc, (int)GetTickCount64 () + cnt);
    cnt++;
  }

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
The Moving() method implements moving individuals in a population within the framework of a neuron-like optimization algorithm. Let's break it down step by step:

Initialization of initial values: if 'revision' is 'false', the method performs the first initialization. For each element in the popSize population and for each coords dimension, the method sets the initial values in the a[i].c[c] array representing the search agents:

u.RNDfromCI () generates a random value in the rangeMin[c] and rangeMax[c] range.
u.SeInDiSp() is a function that uses a step change to adjust the value of a [i].c [c] within a valid range.
After initialization is complete, 'revision' is set to 'true' and control returns from the method. 
Updating coordinates: The coordinates of individuals are updated based on the probability of eliteness. For the first popSize - 5 elements of the population, a random check is performed, and if the random number is less than eliteProb, the a [i].c [c] coordinates are set equal to the coordinates of the "cB [c]" elite.

Handling neural network data: arrays for input and output data, target values and errors are created and prepared to the 'coords' size. For each element of the population:

ScaleInp () scales the cB target coordinates and stores them in targVal.
ScaleInp () - scales the current coordinate from the population and stores it in inpData.
nn [i].ForwProp () performs forward propagation on the neural network to produce output.
The error (the difference between the target and the obtained value) is calculated for each coordinate.
nn [i].BackProp (err) performs backpropagation of the error to adjust the weights.
The a [i].c [c] coordinates are updated taking into account the output of the neural network, scaling, and a random sample element.
At the end, the value of a[i].c[c] is normalized by the u.SeInDiSp function.
Thus, the Moving() method is responsible for moving individuals in the model, initializing them at the beginning and updating their positions based on the outputs of neural networks and probabilistic choices. 

//——————————————————————————————————————————————————————————————————————————————
void C_AO_NOA::Moving ()
{
  //----------------------------------------------------------------------------
  if (!revision)
  {
    // Initialize the initial values of anchors in all parallel universes
    for (int i = 0; i < popSize; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [i].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [i].c [c] = u.SeInDiSp  (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      }
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  double rnd  = 0.0;
  double val  = 0.0;
  int    pair = 0.0;

  for (int i = 0; i < popSize - 5; i++)
  {
    for (int c = 0; c < coords; c++)
    {
      if (u.RNDprobab () < eliteProb)
      {
        a [i].c [c] = cB [c];
      }
    }
  }

  double inpData []; ArrayResize (inpData, coords);
  double outData []; ArrayResize (outData, coords);
  double targVal []; ArrayResize (targVal, coords);
  double err     []; ArrayResize (err,     coords);

  for (int i = 0; i < popSize; i++)
  {
    ScaleInp (cB,      targVal);
    ScaleInp (a [i].c, inpData);

    nn [i].ForwProp (inpData, outData);

    for (int c = 0; c < coords; c++) err [c] = targVal [c] - outData [c];

    nn [i].BackProp (err);

    for (int c = 0; c < coords; c++)
    {
      a [i].c [c] += outData [c] * (rangeMax [c] - rangeMin [c]) * dispScale * u.RNDprobab ();
      a [i].c [c] = u.SeInDiSp  (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
Let's next examine the Revision method, which evaluates and updates the current "best" solution in the population. It compares the objective function values for each individual with the current "best" value and updates it if it finds a better one. Thus, the method ensures that the current best solution is maintained in the population. 

//——————————————————————————————————————————————————————————————————————————————
void C_AO_NOA::Revision ()
{
  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > fB)
    {
      fB = a [i].f;
      ArrayCopy (cB, a [i].c);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
The ScaleInp method is responsible for scaling the input data from the range specified by rangeMin and rangeMax to the interval from -1 to 1. Each value of inp[c] is scaled using the u.Scale function, which performs a linear scaling to bring it into the range from -1 to 1. The ScaleInp method prepares data for further calculations, ensuring its normalization to the desired range.

//——————————————————————————————————————————————————————————————————————————————
void C_AO_NOA::ScaleInp (double &inp [], double &out [])
{
  for (int c = 0; c < coords; c++) out [c] = u.Scale (inp [c], rangeMin [c], rangeMax [c], -1, 1);
}
//——————————————————————————————————————————————————————————————————————————————
The ScaleOut method works similarly to ScaleInp, which performs inverse scaling from the range from -1 to 1 into the rangeMin and rangeMax interval.

//——————————————————————————————————————————————————————————————————————————————
void C_AO_NOA::ScaleOut (double &inp [], double &out [])
{
  for (int c = 0; c < coords; c++) out [c] = u.Scale (inp [c], -1, 1, rangeMin [c], rangeMax [c]);
}
//——————————————————————————————————————————————————————————————————————————————


Test results
The algorithm shows interesting results on functions with small and moderate dimensions. When the problem size increases to 1000 variables, the computation time becomes unacceptably large, as a result of which these tests are not presented. The algorithm achieves approximately 45% of the optimal value in 6 out of 9 test cases.

NOA|Neuroboids Optimization Algorithm (joo)|50.0|0.0|0.01|0.1|
=============================
5 Hilly's; Func runs: 10000; result: 0.7013521128826248
25 Hilly's; Func runs: 10000; result: 0.40128968110640306
=============================
5 Forest's; Func runs: 10000; result: 0.6222984295200933
25 Forest's; Func runs: 10000; result: 0.30830340651626337
=============================
5 Megacity's; Func runs: 10000; result: 0.4523076923076924
25 Megacity's; Func runs: 10000; result: 0.20892307692307693
=============================
All score: 2.69447 (44.91%)

I would like to say a few words about the visualization of the NOA algorithm. As you can see, the algorithm forms interesting fan-shaped structures. In essence, these structures are a visualization of the results of the neural networks in neuroboids. 

Hilly

NOA on the Hilly test function

Forest

NOA on the Forest test function



NOA on the Megacity test function

After the tests the NOA algorithm is placed in a separate line (it does not have a serial number), since no testing results were obtained on high-dimensional functions. The tables below are presented for comparative analysis of results with other algorithms participating in the rating table.

#	AO	Description	Hilly	Hilly final	Forest	Forest final	Megacity (discrete)	Megacity final	Final result	% of MAX
10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)	10 p (5 F)	50 p (25 F)	1000 p (500 F)
1	ANS	across neighbourhood search	0.94948	0.84776	0.43857	2.23581	1.00000	0.92334	0.39988	2.32323	0.70923	0.63477	0.23091	1.57491	6.134	68.15
2	CLA	code lock algorithm (joo)	0.95345	0.87107	0.37590	2.20042	0.98942	0.91709	0.31642	2.22294	0.79692	0.69385	0.19303	1.68380	6.107	67.86
3	AMOm	animal migration ptimization M	0.90358	0.84317	0.46284	2.20959	0.99001	0.92436	0.46598	2.38034	0.56769	0.59132	0.23773	1.39675	5.987	66.52
4	(P+O)ES	(P+O) evolution strategies	0.92256	0.88101	0.40021	2.20379	0.97750	0.87490	0.31945	2.17185	0.67385	0.62985	0.18634	1.49003	5.866	65.17
5	CTA	comet tail algorithm (joo)	0.95346	0.86319	0.27770	2.09435	0.99794	0.85740	0.33949	2.19484	0.88769	0.56431	0.10512	1.55712	5.846	64.96
6	TETA	time evolution travel algorithm (joo)	0.91362	0.82349	0.31990	2.05701	0.97096	0.89532	0.29324	2.15952	0.73462	0.68569	0.16021	1.58052	5.797	64.41
7	SDSm	stochastic diffusion search M	0.93066	0.85445	0.39476	2.17988	0.99983	0.89244	0.19619	2.08846	0.72333	0.61100	0.10670	1.44103	5.709	63.44
8	BOAm	billiards optimization algorithm M	0.95757	0.82599	0.25235	2.03590	1.00000	0.90036	0.30502	2.20538	0.73538	0.52523	0.09563	1.35625	5.598	62.19
9	AAm	archery algorithm M	0.91744	0.70876	0.42160	2.04780	0.92527	0.75802	0.35328	2.03657	0.67385	0.55200	0.23738	1.46323	5.548	61.64
10	ESG	evolution of social groups (joo)	0.99906	0.79654	0.35056	2.14616	1.00000	0.82863	0.13102	1.95965	0.82333	0.55300	0.04725	1.42358	5.529	61.44
11	SIA	simulated isotropic annealing (joo)	0.95784	0.84264	0.41465	2.21513	0.98239	0.79586	0.20507	1.98332	0.68667	0.49300	0.09053	1.27020	5.469	60.76
12	ACS	artificial cooperative search	0.75547	0.74744	0.30407	1.80698	1.00000	0.88861	0.22413	2.11274	0.69077	0.48185	0.13322	1.30583	5.226	58.06
13	DA	dialectical algorithm	0.86183	0.70033	0.33724	1.89940	0.98163	0.72772	0.28718	1.99653	0.70308	0.45292	0.16367	1.31967	5.216	57.95
14	BHAm	black hole algorithm M	0.75236	0.76675	0.34583	1.86493	0.93593	0.80152	0.27177	2.00923	0.65077	0.51646	0.15472	1.32195	5.196	57.73
15	ASO	anarchy society optimization	0.84872	0.74646	0.31465	1.90983	0.96148	0.79150	0.23803	1.99101	0.57077	0.54062	0.16614	1.27752	5.178	57.54
16	RFO	royal flush optimization (joo)	0.83361	0.73742	0.34629	1.91733	0.89424	0.73824	0.24098	1.87346	0.63154	0.50292	0.16421	1.29867	5.089	56.55
17	AOSm	atomic orbital search M	0.80232	0.70449	0.31021	1.81702	0.85660	0.69451	0.21996	1.77107	0.74615	0.52862	0.14358	1.41835	5.006	55.63
18	TSEA	turtle shell evolution algorithm (joo)	0.96798	0.64480	0.29672	1.90949	0.99449	0.61981	0.22708	1.84139	0.69077	0.42646	0.13598	1.25322	5.004	55.60
19	DE	differential evolution	0.95044	0.61674	0.30308	1.87026	0.95317	0.78896	0.16652	1.90865	0.78667	0.36033	0.02953	1.17653	4.955	55.06
20	SRA	successful restaurateur algorithm (joo)	0.96883	0.63455	0.29217	1.89555	0.94637	0.55506	0.19124	1.69267	0.74923	0.44031	0.12526	1.31480	4.903	54.48
21	CRO	chemical reaction optimization	0.94629	0.66112	0.29853	1.90593	0.87906	0.58422	0.21146	1.67473	0.75846	0.42646	0.12686	1.31178	4.892	54.36
22	BIO	blood inheritance optimization (joo)	0.81568	0.65336	0.30877	1.77781	0.89937	0.65319	0.21760	1.77016	0.67846	0.47631	0.13902	1.29378	4.842	53.80
23	BSA	bird swarm algorithm	0.89306	0.64900	0.26250	1.80455	0.92420	0.71121	0.24939	1.88479	0.69385	0.32615	0.10012	1.12012	4.809	53.44
24	HS	harmony search	0.86509	0.68782	0.32527	1.87818	0.99999	0.68002	0.09590	1.77592	0.62000	0.42267	0.05458	1.09725	4.751	52.79
25	SSG	saplings sowing and growing	0.77839	0.64925	0.39543	1.82308	0.85973	0.62467	0.17429	1.65869	0.64667	0.44133	0.10598	1.19398	4.676	51.95
26	BCOm	bacterial chemotaxis optimization M	0.75953	0.62268	0.31483	1.69704	0.89378	0.61339	0.22542	1.73259	0.65385	0.42092	0.14435	1.21912	4.649	51.65
27	ABO	african buffalo optimization	0.83337	0.62247	0.29964	1.75548	0.92170	0.58618	0.19723	1.70511	0.61000	0.43154	0.13225	1.17378	4.634	51.49
28	(PO)ES	(PO) evolution strategies	0.79025	0.62647	0.42935	1.84606	0.87616	0.60943	0.19591	1.68151	0.59000	0.37933	0.11322	1.08255	4.610	51.22
29	TSm	tabu search M	0.87795	0.61431	0.29104	1.78330	0.92885	0.51844	0.19054	1.63783	0.61077	0.38215	0.12157	1.11449	4.536	50.40
30	BSO	brain storm optimization	0.93736	0.57616	0.29688	1.81041	0.93131	0.55866	0.23537	1.72534	0.55231	0.29077	0.11914	0.96222	4.498	49.98
31	WOAm	wale optimization algorithm M	0.84521	0.56298	0.26263	1.67081	0.93100	0.52278	0.16365	1.61743	0.66308	0.41138	0.11357	1.18803	4.476	49.74
32	AEFA	artificial electric field algorithm	0.87700	0.61753	0.25235	1.74688	0.92729	0.72698	0.18064	1.83490	0.66615	0.11631	0.09508	0.87754	4.459	49.55
33	AEO	artificial ecosystem-based optimization algorithm	0.91380	0.46713	0.26470	1.64563	0.90223	0.43705	0.21400	1.55327	0.66154	0.30800	0.28563	1.25517	4.454	49.49
34	ACOm	ant colony optimization M	0.88190	0.66127	0.30377	1.84693	0.85873	0.58680	0.15051	1.59604	0.59667	0.37333	0.02472	0.99472	4.438	49.31
35	BFO-GA	bacterial foraging optimization - ga	0.89150	0.55111	0.31529	1.75790	0.96982	0.39612	0.06305	1.42899	0.72667	0.27500	0.03525	1.03692	4.224	46.93
36	SOA	simple optimization algorithm	0.91520	0.46976	0.27089	1.65585	0.89675	0.37401	0.16984	1.44060	0.69538	0.28031	0.10852	1.08422	4.181	46.45
37	ABHA	artificial bee hive algorithm	0.84131	0.54227	0.26304	1.64663	0.87858	0.47779	0.17181	1.52818	0.50923	0.33877	0.10397	0.95197	4.127	45.85
38	ACMO	atmospheric cloud model optimization	0.90321	0.48546	0.30403	1.69270	0.80268	0.37857	0.19178	1.37303	0.62308	0.24400	0.10795	0.97503	4.041	44.90
39	ADAMm	adaptive moment estimation M	0.88635	0.44766	0.26613	1.60014	0.84497	0.38493	0.16889	1.39880	0.66154	0.27046	0.10594	1.03794	4.037	44.85
40	CGO	chaos game optimization	0.57256	0.37158	0.32018	1.26432	0.61176	0.61931	0.62161	1.85267	0.37538	0.21923	0.19028	0.78490	3.902	43.35
41	ATAm	artificial tribe algorithm M	0.71771	0.55304	0.25235	1.52310	0.82491	0.55904	0.20473	1.58867	0.44000	0.18615	0.09411	0.72026	3.832	42.58
42	ASHA	artificial showering algorithm	0.89686	0.40433	0.25617	1.55737	0.80360	0.35526	0.19160	1.35046	0.47692	0.18123	0.09774	0.75589	3.664	40.71
43	ASBO	adaptive social behavior optimization	0.76331	0.49253	0.32619	1.58202	0.79546	0.40035	0.26097	1.45677	0.26462	0.17169	0.18200	0.61831	3.657	40.63
44	MEC	mind evolutionary computation	0.69533	0.53376	0.32661	1.55569	0.72464	0.33036	0.07198	1.12698	0.52500	0.22000	0.04198	0.78698	3.470	38.55
45	CSA	circle search algorithm	0.66560	0.45317	0.29126	1.41003	0.68797	0.41397	0.20525	1.30719	0.37538	0.23631	0.10646	0.71815	3.435	38.17
NOA	neuroboids optimization algorithm (joo)	0.70135	0.40129	0.00000	1.10264	0.62230	0.30830	0.00000	0.93060	0.45231	0.20892	0.00000	0.66123	2.694	29.94
RW	random walk	0.48754	0.32159	0.25781	1.06694	0.37554	0.21944	0.15877	0.75375	0.27969	0.14917	0.09847	0.52734	2.348	26.09
 

Summary
In creating NOA, I aimed to combine two worlds: neural networks and optimization algorithms. The result was both predictable and unexpected.

The main thing that inspires me about this algorithm is its biological plausibility. We often try to create complex models while nature has been demonstrating efficient solutions for millions of years through the collective intelligence of simple organisms. Neuroboids are an attempt to embody this natural wisdom in algorithmic form.

Testing on various functions showed promising results. About 45% efficiency is a decent start for a conceptually new approach. I see that the algorithm performs well in tasks of medium complexity. However, there are also significant limitations. Scalability issues become critical when the problem size is high. At 1000 variables, the computational load becomes impractical due to the need to train multiple neural networks, a process that is itself resource intensive.

What particularly appeals to me about NOA is its ability to self-learn search strategies. Unlike conventional metaheuristics with fixed rules, neuroboids adapt their behavior through learning.

The variability in results between runs that I observed is twofold. On the one hand, it is a lack of predictability. On the other hand, it is a sign of diversity in exploration strategies, which can be an advantage in complex optimization landscapes.

I see several directions for the development of the NOA algorithm:

Optimizing neural network architecture to reduce computational load
Implementation of knowledge transfer mechanisms between neuroboids
Ultimately, NOA is not just another optimization method. This is a step toward understanding how simple systems can generate complex behavior. This is an exploration of the boundary between machine learning and metaheuristic optimization, between individual and collective intelligence.

I believe this approach has a future not only in function optimization, but also in broader areas – from modeling adaptive behavior to finding fundamentally new artificial intelligence architectures. In the world where complexity is becoming the norm, sometimes simplicity, when properly organized, can yield unexpectedly interesting solutions.

Overall, the NOA algorithm should not be seen as a complete solution to optimization problems, but rather as a basic platform and starting point for many areas of research and the creation of promising solutions in the field of optimization in general and machine learning in particular.

Tab

Figure 2. Color gradation of algorithms according to the corresponding tests

Chart

Figure 3. Histogram of algorithm testing results (scale from 0 to 100, the higher the better, where 100 is the maximum possible theoretical result, in the archive there is a script for calculating the rating table)

NOA pros and cons:

Pros:

Simple implementation.
Interesting results.
Cons:

Due to the long execution time, results for multidimensional spaces were not obtained.
The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

github: https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5

Programs used in the article
#	Name	Type	Description
1	#C_AO.mqh
Include
Parent class of population optimization
algorithms
2	#C_AO_enum.mqh
Include
Enumeration of population optimization algorithms
3
MLPa.mqh	Include	MLP neural network with ADAM
4
TestFunctions.mqh
Include
Library of test functions
5
TestStandFunctions.mqh
Include
Test stand function library
6
Utilities.mqh
Include
Library of auxiliary functions
7
CalculationTestResults.mqh
Include
Script for calculating results in the comparison table
8
Testing AOs.mq5
Script	The unified test stand for all population optimization algorithms
9
Simple use of population optimization algorithms.mq5
Script
A simple example of using population optimization algorithms without visualization
10
Test_AO_NOA.mq5
Script	NOA test stand


READ 13