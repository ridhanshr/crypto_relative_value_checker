from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


@dataclass(frozen=True)
class CheckerConfig:
    n_long: int = 2
    n_short: int = 2
    gross_exposure: float = 2.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0005
    initial_equity: float = 100_000.0
    funding_positive_paid_by_long: bool = True


def _validate(data):
    required = {"timestamp", "asset", "price", "signal"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    if data.duplicated(["timestamp", "asset"]).any():
        raise ValueError("Duplicate (timestamp, asset) rows")
    if (data["price"] <= 0).any() or data["price"].isna().any():
        raise ValueError("Price must be positive and non-null")
    if data["signal"].isna().any() or ~data["signal"].map(pd.api.types.is_number).all():
        raise ValueError("Signal must be numeric and non-null")
    return data.sort_values(["timestamp", "asset"]).reset_index(drop=True)


def check_strategy(data, config=None):
    cfg = config or CheckerConfig()
    data = _validate(data)
    if cfg.n_long < 1 or cfg.n_short < 1 or cfg.initial_equity <= 0:
        raise ValueError("Invalid checker configuration")
    timestamps = list(data["timestamp"].drop_duplicates())
    positions = {}
    ranking_rows, position_rows, trade_rows, pnl_rows = [], [], [], []
    equity = cfg.initial_equity
    prev_prices, prev_positions = {}, {}
    for ts in timestamps:
        cross = data[data.timestamp == ts].sort_values(["signal", "asset"], ascending=[False, True])
        if len(cross) < cfg.n_long + cfg.n_short:
            raise ValueError(f"Not enough assets at {ts}")
        longs = cross.head(cfg.n_long)
        shorts = cross.tail(cfg.n_short)
        long_assets, short_assets = set(longs.asset), set(shorts.asset)
        if long_assets & short_assets:
            raise AssertionError("Long/short ranking overlap")
        w = cfg.gross_exposure / 2
        positions = {a: w / cfg.n_long for a in long_assets}
        positions.update({a: -w / cfg.n_short for a in short_assets})
        ranking_rows.append({"timestamp": ts, "long_assets": ",".join(sorted(long_assets)), "short_assets": ",".join(sorted(short_assets)), "long_count": len(longs), "short_count": len(shorts), "rank_overlap": 0})
        current_prices = dict(zip(cross.asset, cross.price))
        for asset, weight in positions.items():
            position_rows.append({"timestamp": ts, "asset": asset, "weight": weight, "price": current_prices[asset]})
        if prev_positions:
            traded = sum(abs(positions.get(a, 0) - prev_positions.get(a, 0)) for a in set(positions) | set(prev_positions)) * equity
            price_pnl = sum(prev_positions.get(a, 0) * equity * (current_prices[a] / prev_prices[a] - 1) for a in prev_positions if a in current_prices and a in prev_prices)
            funding_rates = dict(zip(cross.asset, cross["funding_rate"] if "funding_rate" in cross else [0.0] * len(cross)))
            funding = sum((-prev_positions.get(a, 0) * equity * funding_rates.get(a, 0)) for a in prev_positions)
        else:
            traded = price_pnl = funding = 0.0
        fee = traded * cfg.fee_rate
        slippage = traded * cfg.slippage_rate
        total = price_pnl + funding - fee - slippage
        equity += total
        pnl_rows.append({"timestamp": ts, "price_pnl": price_pnl, "funding_pnl": funding, "fee_cost": fee, "slippage_cost": slippage, "total_pnl": total, "equity": equity, "return": total / (equity - total) if equity != total else 0, "turnover": traded / (equity - total) if equity != total else 0})
        for asset in set(positions) | set(prev_positions):
            trade_rows.append({"timestamp": ts, "asset": asset, "weight_change": positions.get(asset, 0) - prev_positions.get(asset, 0), "notional": abs(positions.get(asset, 0) - prev_positions.get(asset, 0)) * (equity - total)})
        prev_positions, prev_prices = positions, current_prices
    ranking = pd.DataFrame(ranking_rows)
    exposure = pd.DataFrame([{"timestamp": row["timestamp"], "long_exposure": sum(max(v, 0) for v in row_positions.values()), "short_exposure": sum(abs(min(v, 0)) for v in row_positions.values()), "gross_exposure": sum(abs(v) for v in row_positions.values()), "net_exposure": sum(row_positions.values())} for row, row_positions in zip(ranking_rows, [{a: (cfg.gross_exposure / 2 / cfg.n_long if a in set(r["long_assets"].split(",")) else -cfg.gross_exposure / 2 / cfg.n_short) for a in r["long_assets"].split(",") + r["short_assets"].split(",")} for r in ranking_rows])])
    turnover = pd.DataFrame(pnl_rows)
    return {"ranking": ranking, "exposure": exposure, "turnover": turnover[["timestamp", "turnover"]], "trades": pd.DataFrame(trade_rows), "pnl": pd.DataFrame(pnl_rows), "violations": pd.DataFrame()}


def write_reports(result, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(Path(output_dir) / f"{name}.csv", index=False)
    summary = {name: frame.tail(1).to_dict("records") if isinstance(frame, pd.DataFrame) and not frame.empty else [] for name, frame in result.items()}
    (Path(output_dir) / "summary.json").write_text(json.dumps(summary, default=str, indent=2), encoding="utf-8")
