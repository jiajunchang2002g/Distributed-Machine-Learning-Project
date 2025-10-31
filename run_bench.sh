#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: Expected exactly one argument."
    echo ""
    echo "Usage: $0 <CONFIG>"
    echo ""
    echo "Available CONFIG options:"
    echo "  1 - Run bench_1 on [i7-7700*1 & i7-13700*1], 24 tasks, 2 nodes"
    echo "  2 - Run bench_2 on [w5-3423*1 & i7-9700*1], 32 tasks, 2 nodes"
    echo "  3 - Run bench_3 on [dxs-4114*2], 80 tasks, 2 nodes"
    echo "  4 - Run bench_4 on [i7-9700*1 & i7-13700*1], 24 tasks, 2 nodes"
    echo "  all - Run all above configurations "
    echo ""
    echo "Example:"
    echo "  $0 2"
    echo ""
    exit 1
fi

CONFIG=$1

if [[ "$CONFIG" != "all" && "$CONFIG" != "1" && "$CONFIG" != "2" && "$CONFIG" != "3" && "$CONFIG" != "4" ]]; then
    echo "Error: Invalid CONFIG option '$CONFIG'"
    echo "Run '$0' without arguments for usage information."
    exit 1
fi

compare_times() {
    local bench_err=$1
    local engine_err=$2

    if [[ ! -f "$bench_err" || ! -f "$engine_err" ]]; then
        echo "Error: Missing .err files for comparison ($bench_err or $engine_err not found)."
        return
    fi

    local bench_time
    local engine_time
    bench_time=$(grep -oP 'Time taken:\s*\K[0-9]+' "$bench_err")
    engine_time=$(grep -oP 'Time taken:\s*\K[0-9]+' "$engine_err")

    if [[ -z "$bench_time" || -z "$engine_time" ]]; then
        echo "Error: Could not extract timing information from .err files."
        return
    fi

    echo ""
    echo "=== Performance Comparison ==="
    echo "Benchmark time: ${bench_time} ms"
    echo "Engine time:    ${engine_time} ms"

    local diff=$((engine_time - bench_time))
    local abs_diff=${diff#-}

    if [ "$bench_time" -ne 0 ]; then
        local percent
        percent=$(awk "BEGIN {printf \"%.2f\", ($engine_time - $bench_time) / $bench_time * 100}")

        if (( $(echo "$percent > 0" | bc -l) )); then
            echo "Difference:     +${abs_diff} ms (${percent}% slower)"
        elif (( $(echo "$percent < 0" | bc -l) )); then
            local faster_percent=${percent#-}
            echo "Difference:     -${abs_diff} ms (${faster_percent}% faster) 🎉🎉🎉"
        else
            echo "Difference:     0 ms (No difference)"
        fi
    fi

    echo "=============================="
    echo ""
}

make
mkdir -p outputs

if [ "$CONFIG" = "1" ]; then
    salloc --constraint="[i7-7700*1&i7-13700*1]" --ntasks=24 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_1.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_1 < inputs/input1.in 2> outputs/test_1.err > outputs/test_1.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input1.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_1.err" "outputs/tmp.err"
fi

if [ "$CONFIG" = "2" ]; then
    salloc --constraint="[w5-3423*1&i7-9700*1]" --ntasks=32 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_2.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_2 < inputs/input2.in 2> outputs/test_2.err > outputs/test_2.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input2.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_2.err" "outputs/tmp.err"
fi

if [ "$CONFIG" = "3" ]; then
    salloc --constraint="[dxs-4114*2]" --ntasks=80 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_3.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_3 < inputs/input2.in 2> outputs/test_3.err > outputs/test_3.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input2.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_3.err" "outputs/tmp.err"
fi

if [ "$CONFIG" = "4" ]; then
    salloc --constraint="[i7-9700*1&i7-13700*1]" --ntasks=24 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_4.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_4 < inputs/input3.in 2> outputs/test_4.err > outputs/test_4.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input3.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_4.err" "outputs/tmp.err"
fi

if [ "$CONFIG" = "all" ]; then
    salloc --constraint="[i7-7700*1&i7-13700*1]" --ntasks=24 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_1.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_1 < inputs/input1.in 2> outputs/test_1.err > outputs/test_1.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input1.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_1.err" "outputs/tmp.err"
    salloc --constraint="[w5-3423*1&i7-9700*1]" --ntasks=32 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_2.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_2 < inputs/input2.in 2> outputs/test_2.err > outputs/test_2.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input2.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_2.err" "outputs/tmp.err"
    salloc --constraint="[dxs-4114*2]" --ntasks=80 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_3.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_3 < inputs/input2.in 2> outputs/test_3.err > outputs/test_3.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input2.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_3.err" "outputs/tmp.err"
    salloc --constraint="[i7-9700*1&i7-13700*1]" --ntasks=24 -N 2 --exclusive --time=00:10:00 bash -c '
        if [[ -f outputs/test_4.err ]]; then
            echo "Output found in cache. Skipping..."
        else
            mpirun --timeout 300 --bind-to hwthread ./benchmarks/bench_4 < inputs/input3.in 2> outputs/test_4.err > outputs/test_4.out
        fi
        mpirun --timeout 300 --bind-to hwthread ./engine < inputs/input3.in 2> outputs/tmp.err > outputs/tmp.out
    '
    compare_times "outputs/test_4.err" "outputs/tmp.err"
fi
