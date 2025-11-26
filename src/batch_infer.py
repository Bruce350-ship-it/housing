import argparse
import os
import sys
import pandas as pd
import joblib
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
	with open(config_path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)


def main() -> int:
	parser = argparse.ArgumentParser(description="Batch inference for housing price model")
	parser.add_argument("--input", required=True, help="Path to input CSV")
	parser.add_argument("--output", required=True, help="Path to output CSV (will be overwritten)")
	parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
	args = parser.parse_args()

	cfg = load_config(args.config)
	model_path = cfg["paths"]["best_model_pickle"]

	if not os.path.exists(args.input):
		print(f"Input CSV not found: {args.input}", file=sys.stderr)
		return 2
	if not os.path.exists(model_path):
		print(f"Model not found: {model_path}", file=sys.stderr)
		return 3

	df = pd.read_csv(args.input)
	pipeline = joblib.load(model_path)
	preds = pipeline.predict(df)
	out = df.copy()
	out["predicted_price"] = preds
	out.to_csv(args.output, index=False)
	print(f"Wrote predictions to {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


