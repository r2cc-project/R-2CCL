#!/usr/bin/env bash

DEVICES=(0 2 3)

declare -A TX_START
declare -A RX_START

echo "Initializing counters..."
for i in "${DEVICES[@]}"; do
    DEV="mlx5_$i"
    PATH_TX="/sys/class/infiniband/$DEV/ports/1/counters/port_xmit_data"
    PATH_RX="/sys/class/infiniband/$DEV/ports/1/counters/port_rcv_data"

    if [ -f "$PATH_TX" ]; then
        TX_START["$DEV"]=$(cat "$PATH_TX")
        RX_START["$DEV"]=$(cat "$PATH_RX")
    fi
    done

while true; do
    printf "\033[H\033[J"
    echo "Interface    TX (MB)       RX (MB)"
    echo "------------------------------------"
    for i in "${DEVICES[@]}"; do
        DEV="mlx5_$i"
        PATH_TX="/sys/class/infiniband/$DEV/ports/1/counters/port_xmit_data"
        PATH_RX="/sys/class/infiniband/$DEV/ports/1/counters/port_rcv_data"

        if [ -f "$PATH_TX" ]; then
            TX_NOW=$(cat "$PATH_TX")
            RX_NOW=$(cat "$PATH_RX")

            TX_DIFF=$((TX_NOW - TX_START["$DEV"]))
            RX_DIFF=$((RX_NOW - RX_START["$DEV"]))

            # Handle counter wrap/reset
            if (( TX_DIFF < 0 )); then
                TX_START["$DEV"]=$TX_NOW
                TX_DIFF=0
            fi
            if (( RX_DIFF < 0 )); then
                RX_START["$DEV"]=$RX_NOW
                RX_DIFF=0
            fi

            TX_TOTAL=$(( TX_DIFF * 4 / 1024 / 1024 ))
            RX_TOTAL=$(( RX_DIFF * 4 / 1024 / 1024 ))

            printf "%-12s %-12s %-12s\n" "$DEV" "$TX_TOTAL" "$RX_TOTAL"
        else
            printf "%-12s %-12s\n" "$DEV" "Not Found"
        fi
    done
    sleep 1
done
