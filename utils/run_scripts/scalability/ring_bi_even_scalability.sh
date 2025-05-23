#!/bin/bash

totalNodes=(16 36 64 100 144 196 256)

outdir=$SIMHOME/HPCA2024/scalability/ring_bi
mkdir -p $outdir

for i in ${!totalNodes[@]}; do
  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs ${totalNodes[$i]} \
      --num-vaults 16 \
      --mini-batch-size $((16*totalNodes[$i])) \
      --network $cnndir/alexnet.csv \
      --run-name "scalability" \
      --outdir $outdir \
      --allreduce ring_bi \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_${totalNodes[$i]}_200.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size $((96000*totalNodes[$i])) \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/scalability_${totalNodes[$i]}_ring_bi_error.log 2>&1 &
done

wait