# Chess Clock vs. Move Time: A Statistical Analysis (Poster Edition)

## 1. Distribution of Thinking Time
The first step in any behavioral analysis is understanding the underlying distribution. Chess move times are highly heavy-tailed. Natural log-transformation ($\ln$) is essential to normalize the variance; without it, the "heavy tail" of long thinks would dominate any statistical estimate.

![Move Time Distribution](file:///home/hl4291/chess_analysis/src/figures/move_time_distribution.png)

---

## 2. The Statistical Framework: Resource Allocation
To isolate the relationship between clock availability and thinking time, we utilize a residual-based approach. By operating in natural log-space, our residuals represent **percentage deviations** from a baseline.

### The Baseline Model
For a move by player $i$ at ply $p$, the observed move time $T$ is modeled as:
$$\ln(T) = \alpha + \beta \ln(\text{Clock Time}) + \gamma_{p} + \delta_{i} + \epsilon$$
where:
- $\text{Clock Time}$ is the time remaining on the clock.
- $\gamma_{p}$ is the **Ply Control** (stage-of-game baseline).
- $\delta_{i}$ is the **Player Control** (intrinsic speed baseline).
- $\epsilon$ is the remaining variance.

### Residual Calculation
We isolate the "Thinking Resource" signal by subtracting the expected medians:
$$\text{Residual} = \ln(T) - \ln(\text{med}_{ply}) - \ln(\text{med}_{player})$$
In this space, a value of $+0.7$ means the player spent $e^{0.7} \approx 2\times$ their typical time for that ply and speed profile.

---

## 3. The Progression of Controls
We present the analysis in a standardized **Quad-View Poster** format, with top/right borders removed and subgrids enabled for clarity.

### Attempt 1: The Naive Aggregate (Blue)
The raw data shows a shallow positive trend ($\beta \approx 0.09$), but is corrupted by opening theory, where players have a full clock but move instantly.

![Naive Trend](file:///home/hl4291/chess_analysis/src/figures/attempt1_naive_trend.png)

### Attempt 2: The Ply-Controlled Paradox (Red)
Removing the game stage effect ($\gamma_p$) reveals a paradox: while the binned trends look flat or positive, the individual **Within-Ply Slopes** are negative. This is because faster players (who always have more clock) dominate the high-clock buckets.

![Ply-wise Quad-View](file:///home/hl4291/chess_analysis/src/figures/attempt2_ply_wise_trend.png)

### Attempt 3: The Resolved "Thinking Hypothesis" (Green)
By controlling for **both** player identity ($\delta_i$) and game stage ($\gamma_p$), the paradox is resolved. The positive relationship emerges clearly in all dimensions.

**Result**: The isolated causal effect of the clock is:
$$\frac{\partial \ln(T)}{\partial \ln(\text{Clock Time})} \approx 0.54$$
A 10% increase in clock leads to a ~5.4% increase in thinking time.

![Double-Controlled Quad-View](file:///home/hl4291/chess_analysis/src/figures/attempt3_controlled_trend.png)

---

## 4. Conclusion: The Elasticity of Thinking
This model treats the chess clock as a literal **budget**. The positive coefficient confirms that players are rational resource allocators. The elasticity of thinking time with respect to the clock is approximately 0.54, providing a quantitative baseline for the "Thinking Hypothesis."
