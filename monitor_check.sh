#!/bin/bash
cd /home/user/tihou
while true; do
  sleep 1800
  TIMESTAMP=$(date '+%H:%M')
  if pgrep -f "oi_tune.py" > /dev/null; then
    STATUS="OK"
  else
    STATUS="RESTARTED"
    nohup bash run_tuning_loop.sh > /tmp/tune.log 2>&1 &
  fi
  STATS=$(python -c "
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
s = optuna.load_study(study_name='oi_top3_top5_coverage_v2', storage='sqlite:///models/saved/oi_optuna.db')
print(f'trials={len(s.trials)}, best={s.best_value:.4f}')
" 2>/dev/null)
  if git diff --quiet models/saved/oi_optuna.db 2>/dev/null; then
    COMMIT="no-change"
  else
    git add models/saved/oi_optuna.db models/saved/oi_best_params.json 2>/dev/null
    git commit -m "chore: update Optuna DB (auto-checkpoint)" --quiet 2>/dev/null
    git push --quiet 2>/dev/null
    COMMIT="committed+pushed"
  fi
  echo "[$TIMESTAMP] process=$STATUS $STATS git=$COMMIT"
done
