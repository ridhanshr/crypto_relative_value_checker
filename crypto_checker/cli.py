import argparse
import pandas as pd
from .core import CheckerConfig, check_strategy, write_reports

parser = argparse.ArgumentParser(description="Audit cross-sectional crypto strategy")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="reports")
args = parser.parse_args()
result = check_strategy(pd.read_csv(args.input), CheckerConfig())
write_reports(result, args.output)
print(f"Reports written to {args.output}")
