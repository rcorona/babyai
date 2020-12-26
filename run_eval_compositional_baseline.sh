V=$1

python scripts/evaluate.py --model babyai_baseline_short --results_path results/babyai_baseline_short/compositional.pkl --episodes 1000 --env BabyAI-BossLevel-v0 BabyAI-GoToImpUnlock-v0 BabyAI-Synth-v0 BabyAI-SynthLoc-v0 BabyAI-SynthSeq-v0 
