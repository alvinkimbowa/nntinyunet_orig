#!/bin/bash

# Initialize conda for this script
eval "$(conda shell.bash hook)"

model_name="nnunet"
monogenic=false
if [[ $model_name == "LightMUNet" ]]; then
    conda activate lightmunet
else
    conda activate nnunet
fi

resenc=false

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results/${model_name}"
export nnUNet_compile=false

gpu_id=1
preprocess=1
train=1
predict=0
analyze_model=0
run_inference=0
train_dataset_id=300
test_dataset_ids=(300) #72 73 70 78 79) #8 70 79) #72 73 70 78)
input_dir="data/nnUNet_raw/Dataset079_KneeUS_Ilker/imagesTs"
output_dir="data/nnUNet_raw/Dataset079_KneeUS_Ilker/labelsTs_nnunet_pred"
post_process=0
edge_loss=0
custom_splits=false
modified_plans=false
PLANS_ID=12
fold="0" #$SLURM_ARRAY_TASK_ID
edge_loss=0
# trainer="nnUNetTrainer"
trainer="nnUNetTrainer_20epochs"
# trainer="MonoUNetTrainer"
# trainer="MonoUNetTrainerAdamW"

cfgs=(
    "2d"
    "2d_tiny1"
    "2d_tiny2"
    "2d_tiny4"
    "2d_tiny8"
    "2d_tiny16"
    # "2d_tiny32"
    # "2d_tiny64"
    # "2d_tiny128"
    # "2d_tiny256"
)

# inference parameters
save_preds=false
chk="checkpoint_final.pth"
split="Ts"

overwrite=true    
ensemble=false
largest_component=true

if [[ $fold -eq 5 ]]; then
    fold="all"
fi

echo "preprocess: $preprocess"
echo "train: $train"
echo "predict: $predict"
echo "post_process: $post_process"
echo "train_dataset_id: $train_dataset_id"
echo "test_dataset_ids: $test_dataset_ids"
echo "fold: $fold"
echo "model_name: $model_name"
echo "gpu_id: $gpu_id"
echo "resenc: $resenc"
echo "monogenic: $monogenic"
echo "cfg: $cfg"

export CUDA_VISIBLE_DEVICES=$gpu_id

if [ $resenc == true ]; then
    planner="nnUNetPlannerResEncM"
    plans="nnUNetResEncUNetMPlans"
else
    # planner="ExperimentPlanner"
    planner="TinyExperimentPlanner"
    plans="nnUNetPlans"
fi

if [ $edge_loss -eq 1 ]; then
    trainer="nnUNetTrainerEdgeLoss"
elif [[ $model_name == "attn_unet" ]]; then
    trainer="AttentionUNetTrainer"
elif [[ $model_name == "unet++" ]]; then
    trainer="UNetPlusPlusTrainer"
elif [[ $model_name == "SegResNet" ]]; then
    if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
        trainer="SegResNetTrainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
        trainer="SegResNetTiny1Trainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
        trainer="SegResNetTiny2Trainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
        trainer="SegResNetTiny4Trainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then
        trainer="SegResNetXTiny1Trainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then
        trainer="SegResNetXTiny2Trainer"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
        trainer="SegResNetXTiny4Trainer"
    fi
elif [[ $model_name == "UNETR" ]]; then
    trainer="UNETRTrainer"
elif [[ $model_name == "SwinUNETR" ]]; then
    trainer="SwinUNETRTrainer"
elif [[ $model_name == "UNeXt" ]]; then
    trainer="UNeXtTrainer_S"
elif [[ $model_name == "LightMUNet" ]]; then
    trainer="nnUNetTrainerLightMUNet"
elif [[ $model_name == "CMUNeXt" ]]; then
    trainer="CMUNeXtTrainer_S1"
elif [[ $model_name == "TinyUNet" ]]; then
    trainer="TinyUNetTrainer_S16"
elif [[ $model_name == "MedNCA" ]]; then
    trainer="MedNCATrainerOrig"
fi

if [ $custom_splits == true ]; then
    if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
        trainer=$trainer"_5PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
        trainer=$trainer"_10PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
        trainer=$trainer"_20PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
        trainer=$trainer"_30PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then
        trainer=$trainer"_40PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then
        trainer=$trainer"_50PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
        trainer=$trainer"_60PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 7 ]]; then
        trainer=$trainer"_70PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then
        trainer=$trainer"_80PercentSplit"
    elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then
        trainer=$trainer"_90PercentSplit"
    fi
fi

if [ $modified_plans == true ]; then
    if [[ $PLANS_ID -eq 1 ]]; then
        plans="${plans}_tiny_1"
    elif [[ $PLANS_ID -eq 2 ]]; then
        plans="${plans}_tiny_2"
    elif [[ $PLANS_ID -eq 3 ]]; then
        plans="${plans}_tiny_4"
    elif [[ $PLANS_ID -eq 4 ]]; then
        plans="${plans}_tiny_8"
    elif [[ $PLANS_ID -eq 5 ]]; then
        plans="${plans}_tiny_16"
    elif [[ $PLANS_ID -eq 6 ]]; then
        plans="${plans}_xtiny_1"
    elif [[ $PLANS_ID -eq 7 ]]; then
        plans="${plans}_xtiny_2"
    elif [[ $PLANS_ID -eq 8 ]]; then
        plans="${plans}_xtiny_4"
    elif [[ $PLANS_ID -eq 9 ]]; then
        plans="${plans}_xtiny_8"
    elif [[ $PLANS_ID -eq 10 ]]; then
        plans="${plans}_xtiny_16"
    elif [[ $PLANS_ID -eq 11 ]]; then
        plans="${plans}_xtiny_32"
    elif [[ $PLANS_ID -eq 12 ]]; then
        plans="${plans}_xtiny_64"
    elif [[ $PLANS_ID -eq 13 ]]; then
        plans="${plans}_inverted_8"
    elif [[ $PLANS_ID -eq 14 ]]; then
        plans="${plans}_inverted_16"
    elif [[ $PLANS_ID -eq 15 ]]; then
        plans="${plans}_inverted_32"
    elif [[ $PLANS_ID -eq 16 ]]; then
        plans="${plans}_inverted_8"
    elif [[ $PLANS_ID -eq 17 ]]; then
        plans="${plans}_inverted_16"
    elif [[ $PLANS_ID -eq 18 ]]; then
        plans="${plans}_inverted_32"
    elif [[ $PLANS_ID -eq 19 ]]; then
        plans="${plans}_inverted_64"
    elif [[ $PLANS_ID -eq 20 ]]; then
        plans="${plans}_inverted_512"
    elif [[ $PLANS_ID -eq 21 ]]; then
        plans="${plans}_1"
    elif [[ $PLANS_ID -eq 22 ]]; then
        plans="${plans}_2"
    elif [[ $PLANS_ID -eq 23 ]]; then
        plans="${plans}_4"
    elif [[ $PLANS_ID -eq 24 ]]; then
        plans="${plans}_8"
    elif [[ $PLANS_ID -eq 25 ]]; then
        plans="${plans}_16"
    elif [[ $PLANS_ID -eq 26 ]]; then
        plans="${plans}_32"
    elif [[ $PLANS_ID -eq 27 ]]; then
        plans="${plans}_64"
    elif [[ $PLANS_ID -eq 28 ]]; then
        plans="${plans}_128"
    elif [[ $PLANS_ID -eq 29 ]]; then
        plans="${plans}_xtiny_1_inverted"
    elif [[ $PLANS_ID -eq 30 ]]; then
        plans="${plans}_xtiny_2_inverted"
    elif [[ $PLANS_ID -eq 31 ]]; then
        plans="${plans}_xtiny_4_inverted"
    elif [[ $PLANS_ID -eq 32 ]]; then
        plans="${plans}_xtiny_8_inverted"
    elif [[ $PLANS_ID -eq 33 ]]; then
        plans="${plans}_xtiny_16_inverted"
    elif [[ $PLANS_ID -eq 34 ]]; then
        plans="${plans}_xtiny_32_inverted"
    fi
fi

echo trainer $trainer
echo plans $plans

if [ $preprocess -eq 1 ]; then
    nnUNetv2_plan_and_preprocess -d $train_dataset_id --verify_dataset_integrity -c 2d -pl $planner
fi

if [ $train -eq 1 ]; then
    nnUNetv2_train $train_dataset_id $cfg $fold -p $plans --c -tr $trainer
fi

if [ $predict -eq 1 ]; then
    if [ $train_dataset_id -eq 70 ]; then
        dataset_name="Dataset070_Clarius_L15"
    elif [ $train_dataset_id -eq 71 ]; then
        dataset_name="Dataset071_Sonix-Touch"
    elif [ $train_dataset_id -eq 72 ]; then
        dataset_name="Dataset072_GE_LQP9"
    elif [ $train_dataset_id -eq 73 ]; then
        dataset_name="Dataset073_GE_LE"
    else
        echo "Invalid train dataset id. Current supported dataset ids are (Dataset70, Dataset71, Dataset72)"
        exit 1
    fi

    nnUNetv2_predict -i $input_dir -o $output_dir -d $train_dataset_id -c $cfg -f 0 -p $plans
fi

if [ $post_process -eq 1 ]; then
    python rename_segmentations.py \
        -p $output_dir \
        -d $train_dataset_id
fi


if [ $run_inference -eq 1 ]; then
    for test_dataset_id in ${test_dataset_ids[@]}; do
        echo test_dataset_id $test_dataset_id
        if [[ $test_dataset_id -eq $train_dataset_id ]]; then
            split="Val"
        elif [[ $test_dataset_id -eq 78 || $test_dataset_id -eq 79 || $test_dataset_id -eq 178 ]]; then
            split="Ts"
        else
            split="Tr"
        fi
        python inference.py \
            --train_dataset_id $train_dataset_id \
            --test_dataset_id $test_dataset_id \
            --save_preds $save_preds \
            --model_name $model_name \
            --chk $chk \
            --split $split \
            --overwrite $overwrite \
            --cfg $cfg \
            --plans $plans \
            --trainer $trainer \
            --fold $fold \
            --ensemble $ensemble \
            --largest_component $largest_component
    done
fi

if [ $analyze_model -eq 1 ]; then
    conda activate clarius_deploy
    if [[ $model_name == "LightMUNet" ]]; then
        conda activate lightmunet
    fi

    python analyze_model.py \
        --train_dataset_id $train_dataset_id \
        --model_name $model_name \
        --plans $plans \
        --trainer $trainer \
        --fold $fold \
        --cfg $cfg \
        --chk $chk \
        --gpu $gpu_id
fi