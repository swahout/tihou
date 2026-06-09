#!/bin/bash
# チューニングを継続的に実行するループ
# 200試行ずつ追加し続ける（Ctrl+C または kill で停止）
cd /home/user/tihou
ROUNDS=0
while true; do
    ROUNDS=$((ROUNDS + 1))
    echo "=== ラウンド $ROUNDS 開始 (200試行) ===" | tee -a /tmp/tune_loop.log
    python -u oi_tune.py --tune --trials 200 2>&1 | tee -a /tmp/tune_loop.log
    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "エラー終了 (exit $EXIT)。30秒後に再試行..." | tee -a /tmp/tune_loop.log
        sleep 30
    else
        echo "=== ラウンド $ROUNDS 完了 ===" | tee -a /tmp/tune_loop.log
    fi
    sleep 5
done
