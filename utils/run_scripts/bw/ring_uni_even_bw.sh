#!/bin/bash

syntheticDataSize=(262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456)
outdir=$SIMHOME/HPCA2024/bandwidth/ring

mkdir -p $outdir

for i in ${!syntheticDataSize[@]}; do
  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 16 \
      --num-vaults 16 \
      --mini-batch-size 256 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce ring \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_16_200.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size ${syntheticDataSize[$i]} \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/bw_${syntheticDataSize[$i]}_ring_16_error.log 2>&1 &

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 64 \
      --num-vaults 16 \
      --mini-batch-size 1024 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce ring \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_64_200.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size ${syntheticDataSize[$i]} \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/bw_${syntheticDataSize[$i]}_ring_64_error.log 2>&1 &
done

wait