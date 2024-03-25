#!/bin/bash

syntheticDataSize=(67108864)
outdir=$SIMHOME/HPCA2024/utilization

mkdir -p $outdir

for i in ${!syntheticDataSize[@]}; do

  # Unidirectional Ring

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1296 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce ring_odd \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
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
      --save-link-utilization \
      > $outdir/bw_${syntheticDataSize[$i]}_ring_81_error.log 2>&1 &

  # Bidirectional Ring

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1296 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce ring_odd_bi \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
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
      --save-link-utilization \
      > $outdir/bw_${syntheticDataSize[$i]}_ring_bi_81_error.log 2>&1 &

  # Hierarchical Ring

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1296 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce ring2dn \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
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
      --save-link-utilization \
      > $outdir/bw_${syntheticDataSize[$i]}_ring2dn_81_error.log 2>&1 &

  # Multitree

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1296 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce multitree \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
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
      --load-tree \
      --save-link-utilization \
      > $outdir/bw_${syntheticDataSize[$i]}_multitree_81_mesh_error.log 2>&1 &

  # TTO

  python $SIMHOME/src/simulate.py \
      --arch-config $SIMHOME/src/SCALE-Sim/configs/google.cfg \
      --num-hmcs 81 \
      --num-vaults 16 \
      --mini-batch-size 1296 \
      --network $cnndir/alexnet.csv \
      --run-name "bw_${syntheticDataSize[$i]}" \
      --outdir $outdir \
      --allreduce mesh_overlap_2d_1 \
      --booksim-config $SIMHOME/src/booksim2/runfiles/mesh/anynet_mesh_81_200.cfg \
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
      --save-link-utilization \
      > $outdir/bw_${syntheticDataSize[$i]}_mesh_overlap_2d_1_81_mesh_error.log 2>&1 &
done

wait