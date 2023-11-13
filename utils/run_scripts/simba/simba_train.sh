#!/bin/bash

outdir=$SIMHOME/HPCA2024/simba/train

mkdir -p $outdir

mlperfdir=$SIMHOME/src/SCALE-Sim/topologies/mlperf
cnndir=$SIMHOME/src/SCALE-Sim/topologies/conv_nets

for nnpath in $mlperfdir/AlphaGoZero $mlperfdir/FasterRCNN $mlperfdir/NCF_recommendation \
  $mlperfdir/Resnet152 $mlperfdir/Transformer \
  $cnndir/alexnet $cnndir/Googlenet
do
  nn=$(basename $nnpath)
  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/express.cfg \
      --num-hmcs 36 \
      --num-vaults 16 \
      --mini-batch-size 576 \
      --network $nnpath.csv \
      --run-name ${nn} \
      --outdir $outdir \
      --allreduce mesh_overlap_2d_1 \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_36_200.cfg \
      --booksim-network mesh \
      --only-compute \
      --message-buffer-size 32 \
      --message-size 8192 \
      --sub-message-size 8192 \
      --synthetic-data-size 0 \
      --flits-per-packet 16 \
      --bandwidth 200 \
      --kary 5 \
      --radix 4 \
      --strict-schedule \
      --prioritize-schedule \
      > $outdir/${nn}_mesh_overlap_2d_1_36_error_express.log 2>&1 &

  python $SIMHOME/src/simulate.py \
    --arch-config $SIMHOME/src/SCALE-Sim/configs/express_16.cfg \
    --num-hmcs 36 \
    --num-vaults 16 \
    --mini-batch-size 576 \
    --network $nnpath.csv \
    --run-name ${nn} \
    --outdir $outdir \
    --allreduce mesh_overlap_2d_1 \
    --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_36_200.cfg \
    --booksim-network mesh \
    --only-compute \
    --message-buffer-size 32 \
    --message-size 8192 \
    --sub-message-size 8192 \
    --synthetic-data-size 0 \
    --flits-per-packet 16 \
    --bandwidth 200 \
    --kary 5 \
    --radix 4 \
    --strict-schedule \
    --prioritize-schedule \
    > $outdir/${nn}_mesh_overlap_2d_1_36_error_express_16.log 2>&1 &
done