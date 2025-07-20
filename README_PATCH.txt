Replace indicator .squeeze() calls with .values.ravel() to fix shape issues.

# Example fix inside scan.py where indicators are calculated
# Old line (which caused shape error):
# rsi = ta.momentum.RSIIndicator(df['Close']).rsi()

# New line:
rsi = ta.momentum.RSIIndicator(df['Close']).rsi().values.ravel()
