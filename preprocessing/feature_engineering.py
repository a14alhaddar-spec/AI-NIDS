from __future__ import annotations

import pandas as pd


def select_feature_columns(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
	if not features:
		return df
	keep = [c for c in features if c in df.columns]
	if not keep:
		return df
	return df[keep]


def ensure_basic_features(df: pd.DataFrame) -> pd.DataFrame:
	if "bytes_per_sec" not in df.columns and "bytes" in df.columns and "flow_duration" in df.columns:
		duration = df["flow_duration"].replace(0, 1)
		df["bytes_per_sec"] = df["bytes"] / duration
	if "packets_per_sec" not in df.columns and "packets" in df.columns and "flow_duration" in df.columns:
		duration = df["flow_duration"].replace(0, 1)
		df["packets_per_sec"] = df["packets"] / duration
	if (
		"src_dst_bytes_ratio" not in df.columns
		and "src_bytes" in df.columns
		and "dst_bytes" in df.columns
	):
		df["src_dst_bytes_ratio"] = df["src_bytes"] / df["dst_bytes"].replace(0, 1)
	if "avg_pkt_size" not in df.columns and "bytes" in df.columns and "packets" in df.columns:
		df["avg_pkt_size"] = df["bytes"] / df["packets"].replace(0, 1)
	return df
