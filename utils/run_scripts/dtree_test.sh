#!/bin/bash

outdir=$SIMHOME/HPCA2024/models/dtre

mkdir -p $outdir
mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

for nnpath in $mlperfdir/AlphaGoZero
do
  nn=$(basename $nnpath)

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1280 \
      --network $nnpath.csv \
      --run-name ${nn} \
      --outdir $outdir \
      --allreduce dtree \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
      --booksim-network mesh \
      --only-allreduce \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size 0 \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 2 \
      --radix 1 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/${nn}_dtree_81_error.log 2>&1 &
done

wait