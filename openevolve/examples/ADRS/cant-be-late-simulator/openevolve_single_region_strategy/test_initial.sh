#!/bin/bash

echo "Testing initial single-region strategy..."

# Test on one trace file
python main.py \
    --strategy=rc_cr_threshold \
    --env=trace \
    --trace-file data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json \
    --task-duration-hours=48 \
    --deadline-hours=52 \
    --restart-overhead-hours=0.2 \
    --output-dir=outputs/single_region_test

echo ""
echo "Baseline rc_cr_threshold test complete. Check outputs/single_region_test/ for results."