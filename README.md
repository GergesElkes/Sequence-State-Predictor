🧠 Sequence State Predictor (Tkinter GUI)

An interactive desktop prediction tool for forecasting the next state in a financial sequence model using a cleaned transition dataset (out.csv).

Built with Tkinter, this application provides a visual and analytical interface over a probabilistic state transition model, including multi-step future path forecasting.

🚀 Overview

This project wraps a trained state transition model into a user-friendly GUI, allowing you to:

Predict the next most likely state
Analyze transition probabilities
Explore multi-step future paths (3–4 steps ahead)
Incorporate timing regimes (EARLY / CLOCK / LATE) into predictions

It is designed as a research & analysis tool, not for live trading execution.

🧠 Core Concept

The system operates on:

A finite set of market states
A transition probability model built from historical sequences
Optional regime memory (3-token timing context)

Prediction is based on:

(current_state + optional_regime) → next_state probabilities
✨ Features
🎯 Next-State Prediction
Displays top probable next states
Shows:
Probability
Transition count
Ranked outputs
🔮 Future Path Forecasting
Predicts sequences of:
Next 3 states
Next 4 states
Based purely on empirical transitions
⏱ Regime-Aware Modeling

Optional regime context using:

EARLY
CLOCK
LATE

Allows:

Context-sensitive predictions
Detection of compression / expansion behavior
📊 Visual Probability Breakdown
Color-coded predictions:
🟢 BULL states
🔴 BEAR states
🔵 Neutral states
Horizontal probability bars
Sorted ranking table
🎛 Interactive Controls
Select current state
Toggle regime usage
Adjust:
Top N predictions
Minimum probability threshold
Use last observed row instantly
Reload dataset without restarting
⚡ Smart Model Handling
Automatically:
Loads CSV
Cleans stitched sequence breaks
Builds transition probabilities
Runs in background thread (non-blocking UI)
🏗️ Project Structure
project/
├── main.py                  # GUI application (this file)
├── predict_next_state.py    # Core model + prediction logic
├── out.csv                  # Training dataset
├── requirements.txt
├── README.md
📂 Data Requirements

The model expects a CSV (out.csv) with:

state_id
state_name
regime (3-token sequence, e.g. EARLY CLOCK LATE)

The dataset should already be cleaned or structured for sequence modeling.

⚙️ Installation
1. Clone the repository
git clone https://github.com/yourusername/sequence-state-predictor.git
cd sequence-state-predictor
2. Install dependencies
pip install -r requirements.txt
▶️ Usage
Run GUI
python main.py
Run quick CLI check (no GUI)
python main.py --check

This will:

Load the model
Print one prediction
Show a sample future path
Use custom CSV
python main.py --csv your_data.csv
🧪 How It Works
1. Model Building
Reads CSV
Removes invalid transitions ("stitched breaks")
Constructs:
Transition counts
Probability distributions
2. Prediction Engine

Uses:

State-only mode or
State + regime mode

Fallback logic applies when exact context is missing.

3. Future Path Expansion

Simulates forward transitions:

Breadth-based probability expansion
Returns most likely paths with:
Combined probability
Occurrence count
📊 Example Insights
Detect high-probability continuation states
Identify compression states (e.g. 0, 4, 8)
Observe regime-driven behavior changes
Explore multi-step structural patterns
⚠️ Limitations
No machine learning (pure statistical model)
No real-time data feed
Depends heavily on dataset quality
Not intended for automated trading
🚧 Future Improvements
Integrate ML-based prediction layer
Add live data streaming
Export predictions to CSV
Add confidence intervals
Improve performance for large datasets
📸 Screenshots

Add screenshots of:

Main prediction dashboard
Probability table
Future path panels

Example:

![Main UI](screenshot.png)
👤 Author

Gerges Elkes
GitHub: https://github.com/GergesElkes
