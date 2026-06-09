#!/bin/bash
# 100試行ごとに6/10の予測を自動実行
cd /home/user/tihou
LAST_MILESTONE=0
DATE="2026/06/10"

while true; do
  sleep 120  # 2分ごとにチェック
  
  TRIALS=$(python -c "
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
s = optuna.load_study(study_name='oi_top3_top5_coverage_v3', storage='sqlite:///models/saved/oi_optuna.db')
print(len([t for t in s.trials if t.value is not None]))
" 2>/dev/null)
  
  if [ -z "$TRIALS" ]; then continue; fi
  
  MILESTONE=$(( (TRIALS / 100) * 100 ))
  
  if [ "$MILESTONE" -gt "$LAST_MILESTONE" ] && [ "$MILESTONE" -ge 100 ]; then
    LAST_MILESTONE=$MILESTONE
    BEST=$(python -c "
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
s = optuna.load_study(study_name='oi_top3_top5_coverage_v3', storage='sqlite:///models/saved/oi_optuna.db')
print(f'{s.best_value:.4f}')
" 2>/dev/null)
    echo "=== ${MILESTONE}試行達成 (best=${BEST}) → 予測実行 ==="
    python oi_tune.py --predict --date "$DATE" 2>&1
    echo "=== 予測完了 ==="
  fi
done
