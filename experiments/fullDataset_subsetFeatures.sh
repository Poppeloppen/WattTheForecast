####################################################################
# Script for testing all the models on ....
####################################################################

# General configurations
DATASET_SIZE=full_dataset
DATASET_FEATURES=subset_features
FEATURES=M
EMBED=fixed
ENC_IN=23
DEC_IN=23
PLOT_FLAG=0
USE_WANDB=1
#N_CLOSEST=None


#########
# Regular models (non-graph)
#########
for MODEL in persistence MLP LSTM FFTransformer
do 
    echo "Running model: $MODEL"
    python -u run.py \
        --model "$MODEL" \
        --embed "$EMBED" \
        --features "$FEATURES" \
        --data Wind \
        --dataset_size "$DATASET_SIZE" \
        --dataset_features "$DATASET_FEATURES" \
        --enc_in "$ENC_IN" \
        --dec_in "$DEC_IN" \
        --plot_flag "$PLOT_FLAG" \
        --use_wandb "$USE_WANDB"
        #--n_closest "$N_CLOSEST"
done


#########
# GNN-based models
#########
for MODEL in GraphPersistence GraphMLP GraphLSTM GraphFFTransformer
do 
    echo "Running model: $MODEL"
    python -u run.py \
        --model "$MODEL" \
        --embed "$EMBED" \
        --features "$FEATURES" \
        --data WindGraph \
        --dataset_size "$DATASET_SIZE" \
        --dataset_features "$DATASET_FEATURES" \
        --enc_in "$ENC_IN" \
        --dec_in "$DEC_IN" \
        --plot_flag "$PLOT_FLAG" \
        --use_wandb "$USE_WANDB"
        #--n_closest "$N_CLOSEST"
done
