#!/bin/bash
# 収集スクリプトが終了するたびに自動再起動するラッパー
# 全場の収集が完了したら自動でv5チューニングを開始する

TIHOU=/home/user/tihou
LOG_DIR=/tmp

restart_count=0

while true; do
    # 船橋
    FUNABASHI_DONE=false
    if [ -f "$TIHOU/data/historical_funabashi/funabashi_2026.csv" ]; then
        FUNABASHI_DONE=true
    fi

    # 浦和
    URAWA_DONE=false
    if [ -f "$TIHOU/data/historical_urawa/urawa_2026.csv" ]; then
        URAWA_DONE=true
    fi

    # チューニング済み
    TUNE_DONE=false
    if [ -f "$LOG_DIR/kawasaki_tune_v5.log" ] && grep -q "Best trial" "$LOG_DIR/kawasaki_tune_v5.log" 2>/dev/null; then
        TUNE_DONE=true
    fi

    if [ "$FUNABASHI_DONE" = false ]; then
        echo "[$(date)] 船橋収集開始 (restart=$restart_count)"
        python "$TIHOU/collect_historical_nankan.py" --venue 船橋 --years 2022 2023 2024 2025 2026 --delay 1.2
        echo "[$(date)] 船橋収集プロセス終了"

    elif [ "$URAWA_DONE" = false ]; then
        echo "[$(date)] 浦和収集開始 (restart=$restart_count)"
        python "$TIHOU/collect_historical_nankan.py" --venue 浦和 --years 2022 2023 2024 2025 2026 --delay 1.2
        echo "[$(date)] 浦和収集プロセス終了"

    elif [ "$TUNE_DONE" = false ]; then
        echo "[$(date)] v5チューニング開始"
        python "$TIHOU/kawasaki_predict.py" --tune --trials 200
        echo "[$(date)] チューニング終了"
        break  # チューニングは1回でOK

    else
        echo "[$(date)] 全タスク完了"
        break
    fi

    restart_count=$((restart_count + 1))
    sleep 5
done
