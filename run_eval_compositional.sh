V=$1

python scripts/evaluate.py --model ${V} --results_path results/${V}/compositional.pkl --episodes 1000 --env BabyAI-BossLevel-v0 BabyAI-GoToImpUnlock-v0 BabyAI-Synth-v0 BabyAI-SynthLoc-v0 BabyAI-SynthSeq-v0
