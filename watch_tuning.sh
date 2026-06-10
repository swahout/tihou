#!/bin/bash
# チューニング監視・自動再起動スクリプト
LOG=/home/user/tihou/watch_tuning.log
OUTDIR=/home/user/tihou
cd $OUTDIR

echo "[$(date)] 監視開始" >> $LOG

while true; do
    # プロセスが生きているか確認
    if pgrep -f "tune_oi_6mo.py --tune" > /dev/null; then
        echo "[$(date)] 実行中" >> $LOG
    else
        # 完了済みか確認（200試行に達した場合は再起動しない）
        BEST=$(python3 -c "
import optuna, warnings
warnings.filterwarnings('ignore')
try:
    s = optuna.load_study(study_name='oi_6mo_top5_coverage_v2', storage='sqlite:///models/saved/oi_6mo_optuna.db')
    print(len(s.trials), s.best_value)
except: print('0 0')
" 2>/dev/null)
        N=$(echo $BEST | awk '{print $1}')
        echo "[$(date)] プロセスなし trials=$N" >> $LOG

        if [ "$N" -ge 200 ] 2>/dev/null; then
            echo "[$(date)] 200試行完了。監視終了。" >> $LOG
            break
        else
            echo "[$(date)] 停止検知 ($N試行) → 再起動" >> $LOG
            python3 -u tune_oi_6mo.py --tune --trials 200 >> /home/user/tihou/tuning_v2.log 2>&1 &
            sleep 10
        fi
    fi
    sleep 180  # 3分ごとにチェック
done
