#!/bin/bash

clusterSize=(9 16 25 36 49 64 81 100 121 144 169 196 225)

outdir=$SIMHOME/HPCA2024/tree/multitree

mkdir -p $outdir

for i in ${!clusterSize[@]}; do
  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/express.cfg \
      --num-hmcs ${clusterSize[$i]} \
      --num-vaults 16 \
      --mini-batch-size 256 \
      --network $cnndir/alexnet.csv \
      --run-name "tree" \
      --outdir $outdir \
      --allreduce multitree \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_2D_4_Ring_Ring_4_4_400_400_mesh.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8480 \
      --sub-message-size 8480 \
      --synthetic-data-size 262144 \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/tree_${clusterSize[$i]}_multitree_mesh_error.log 2>&1 &
done

wait