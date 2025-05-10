###########################################################################
# Script for running experiments on ST-LSTM models (1-, 4- and 24-step)
###########################################################################
# NOTE: some parameters change depending on the forecast-horizon of the model
#       these parameters are the ones defined as tuples below - the first value
#       is used for the 1-step model, the second for the 4-step and the final for 
#       the 24-step model.


#Basic configs
MODEL_ID=
MODEL=GraphLSTM
PLOT_FLAG=0
USE_WANDB=0

#Data loader 
DATA=WindGraph
DATASET_SIZE=full_dataset
DATASET_FEATURES=subset_features
N_CLOSEST=5             #NOTE: could try other values

#Forecasting task
FEATURES=M
SEQ_LENS=(32 32 64)
LABEL_LENS=(24 24 48)    #NOTE: simply use 0.75 of seq_len
PRED_LENS=(1 4 24)
ENC_IN=8
DEC_IN=8
C_OUT=1

#Model defn.
D_MODELS=(16 32 64)
E_LAYERS=(1 1 2)
GNN_LAYERS=2
D_FFS=(64 128 256)     #NOTE: Not sure this is the right parameter...
DROPOUT=(0.05 0.05 0.15)
EMBED=fixed             #NOTE: only embedding implemented
ACTIVATION=relu
MLP_OUT=0               #NOTE: double check that we do not want this - 0 is default tho

#Optimization
ITR=1
TRAIN_EPOCHS=10
BATCH_SIZE=32
PATIENCE=5
LEARNING_RATE=0.001
LR_DECAY_RATE=0.8
LRADJ=type1             #NOTE: check which type to use...

#Hardware acceleration (will automatically use whichever is available)
USE_GPU=1
USE_MPS=1



for i in {0..2}
do  
    
    SEQ_LEN="${SEQ_LENS[$i]}"
    PRED_LEN="${PRED_LENS[$i]}"
    LABEL_LEN="${LABEL_LENS[$i]}"
    
    D_MODEL="${D_MODELS[$i]}"
    D_FF="${D_FFS[$i]}"

    echo "Running experiment with ${PRED_LEN}-step $MODEL"

    python -u run.py \
    --model_id "${PRED_LEN}Step" \
    --model "$MODEL" \
    --plot_flag "$PLOT_FLAG" \
    --use_wandb "$USE_WANDB" \
    --data "$DATA" \
    --dataset_size "$DATASET_SIZE" \
    --dataset_features "$DATASET_FEATURES" \
    --features "$FEATURES" \
    --seq_len "$SEQ_LEN" \
    --label_len "$LABEL_LEN" \
    --pred_len "$PRED_LEN" \
    --enc_in "$ENC_IN" \
    --dec_in "$DEC_IN" \
    --c_out "$C_OUT" \
    --d_model "$D_MODEL" \
    --e_layers "$E_LAYERS" \
    --gnn_layers "$GNN_LAYERS" \
    --d_ff "$D_FF" \
    --dropout "$DROPOUT" \
    --embed "$EMBED" \
    --activation "$ACTIVATION" \
    --mlp_out "$MLP_OUT" \
    --itr "$ITR" \
    --train_epochs "$TRAIN_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --learning_rate "$LEARNING_RATE" \
    --lr_decay_rate "$LR_DECAY_RATE" \
    --lradj "$LRADJ" \
    --use_gpu "$USE_GPU" \
    --use_mps "$USE_MPS" \

done

