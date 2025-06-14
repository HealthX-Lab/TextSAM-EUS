CFG=$1
OUTPUT=$2
DATASET=pancreas_tumors

for SEED in 1
do
    python train.py --config-file configs/${CFG}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}

    python inference_eval_text.py --config-file configs/${CFG}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}

    # python evaluation/eval.py \
    # --gt_path data/${DATASET}/test/masks \
    # --seg_path output/pancreas_tumors/seg_results/seed1/LORA4_SHOTS100_NCTX4_CSCFalse_CTPend \
    # --save_path test.csv 
done