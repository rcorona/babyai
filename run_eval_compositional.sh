V=$1

python scripts/evaluate.py --model_path models/${V}/best.pth --results_path results/${V}/compositional.pkl --model cpv --episodes 1000 --env BabyAI-BossLevel-v0 BabyAI-GoToImpUnlock-v0 BabyAI-Synth-v0 BabyAI-SynthLoc-v0 BabyAI-SynthSeq-v0
