import pandas as pd
import os

df = pd.read_csv("data/scenarios/scenario_benchmark_results.csv")
df_day = df[(df['scenario'] == 'Day_in_the_Life') & (df['orchestrator'] == True)]

# specific seed
df_seed = df_day[df_day['seed'] == 42].sort_values('step')

modes = df_seed['mode'].values
steps = df_seed['step'].values

transitions = []
current = modes[0]
for m in modes:
    if m != current:
        transitions.append(f"{current} -> {m}")
        current = m

print("Mode Transitions:")
for t in transitions:
    print(t)

# Check for Green -> Survival direct jump
direct_jump = False
for t in transitions:
    if "GREEN" in t and "SURVIVAL" in t:
        # Check if it was direct
        # t is string "FROM -> TO"
        parts = t.split(" -> ")
        if "GREEN" in parts[0] and "SURVIVAL" in parts[1]:
             direct_jump = True

if direct_jump:
    print("\n[FAIL] Direct GREEN -> SURVIVAL jump detected!")
else:
    print("\n[PASS] Safety Shield Enforced (No direct jump).")
