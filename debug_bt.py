"""バックテストデバッグ: エラーを全て表示"""
import pandas as pd
import traceback
from pathlib import Path
from models.backtest import _build_race_id, _build_train_features
from models.ensemble import train_ensemble, predict_ensemble
from models.features import build_features

dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in sorted(Path('data/historical_oi').glob('oi_*.csv'))]
df = pd.concat(dfs, ignore_index=True)
df['race_date'] = pd.to_datetime(df['race_date'])
df['year'] = df['race_date'].dt.year
df = df[df['finish_position'].notna()].copy()
df['finish_position'] = df['finish_position'].astype(int)
df['race_id'] = _build_race_id(df)

test_year = 2023
df_train = df[df['year'] < test_year].copy()
df_test  = df[df['year'] == test_year].copy()
print(f'train={len(df_train)}, test={len(df_test)}', flush=True)

print('特徴量構築中...', flush=True)
X_train, y_train, rids = _build_train_features(df_train)
print(f'X_train: {X_train.shape}, NaN合計: {X_train.isna().sum().sum()}', flush=True)
if X_train.empty:
    print('ERROR: X_train is empty!', flush=True)
    exit(1)

print('モデル学習中...', flush=True)
try:
    models = train_ensemble(X_train, y_train, rids)
    print('学習OK', flush=True)
except Exception as e:
    print(f'学習エラー: {e}', flush=True)
    traceback.print_exc()
    exit(1)

print('最初の5レースを評価...', flush=True)
df_test['race_id'] = _build_race_id(df_test)
n_ok = 0
for rid, race_df in df_test.groupby('race_id'):
    if len(race_df) < 4:
        continue
    try:
        X_test = build_features(race_df, df_train)
        scores = predict_ensemble(models, X_test)
        n_ok += 1
        print(f'  {rid}: OK scores={scores[:3].round(3)}', flush=True)
    except Exception as e:
        print(f'  {rid}: ERROR {e}', flush=True)
        traceback.print_exc()
    if n_ok + 1 > 5:
        break

print(f'完了: {n_ok}レース成功', flush=True)
