V=$1

python scripts/evaluate.py --model babyai_baseline_short --results_path results/babyai_baseline_short/atomic.pkl --episodes 1000 --env BabyAI-GoTo-v0 BabyAI-GoToLocal-v0 BabyAI-GoToObj-v0 BabyAI-GoToObjMaze-v0 BabyAI-GoToObjMazeOpen-v0 BabyAI-GoToRedBall-v0 BabyAI-GoToRedBallGrey-v0 BabyAI-GoToSeq-v0 BabyAI-Open-v0 BabyAI-Pickup-v0 BabyAI-PickupLoc-v0 BabyAI-PutNext-v0 BabyAI-PutNextLocal-v0 BabyAI-UnblockPickup-v0 BabyAI-Unlock-v0
