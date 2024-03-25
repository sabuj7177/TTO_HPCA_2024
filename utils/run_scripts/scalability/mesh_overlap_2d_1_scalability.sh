#!/bin/bash

totalNodes=(9 16 25 36 49 64 81 100 121 144 169 196 225 256)

outdir=$SIMHOME/HPCA2024/scalability/mesh_overlap_2d_1
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
      --allreduce mesh_overlap_2d_1 \
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
      > $outdir/scalability_${totalNodes[$i]}_mesh_overlap_2d_1_error.log 2>&1 &
done

wait