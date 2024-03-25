#!/bin/bash

chunkSize=(1536 3072 6144 12288 24576 49152 98304 196608 393216 786432 1572864)
outdir=$SIMHOME/HPCA2024/chunk_utilization
mkdir -p $outdir

for i in ${!chunkSize[@]}; do
  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 64 \
      --num-vaults 16 \
      --mini-batch-size 1024 \
      --network $cnndir/alexnet.csv \
      --run-name "chunk_33554432_${chunkSize[$i]}" \
      --outdir $outdir \
      --allreduce mesh_overlap_2d_1 \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_64_200.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size 33554432 \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      --save-link-utilization \
      --chunk-size ${chunkSize[$i]} \
      > $outdir/bw_33554432_${chunkSize[$i]}_mesh_overlap_2d_1_64_mesh_error.log 2>&1 &
done

wait