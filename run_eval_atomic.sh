V=$1

python scripts/evaluate.py --model_path models/${V}/best.pth --results_path results/${V}/atomic.pkl --model cpv --episodes 1000 --env BabyAI-GoTo-v0 BabyAI-GoToLocal-v0 BabyAI-GoToObj-v0 BabyAI-GoToObjMaze-v0 BabyAI-GoToObjMazeOpen-v0 BabyAI-GoToRedBall-v0 BabyAI-GoToRedBallGrey-v0 BabyAI-GoToSeq-v0 BabyAI-Open-v0 BabyAI-Pickup-v0 BabyAI-PickupLoc-v0 BabyAI-PutNext-v0 BabyAI-PutNextLocal-v0 BabyAI-UnblockPickup-v0 BabyAI-Unlock-v0
