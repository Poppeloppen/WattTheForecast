###########################################################################
# Script for running experiments on the persistence models
###########################################################################


#Only the forecast horizon is changed
PRED_LENS=(1 4 24)


for i in {0..2}
do

    PRED_LEN="${PRED_LENS[$i]}"

    python -u run.py \
        --model_id "Baseline" \
        --model "GraphPersistence" \
        --plot_flag "0" \
        --data "WindGraph" \
        --dataset_size "full_dataset" \
        --dataset_features "subset_features" \
        --features "M" \
        --pred_len "$PRED_LEN" \
        --use_gpu "1" \
        --use_mps "1" \
    
done



