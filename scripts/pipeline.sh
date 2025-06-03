CFG=$1
OUTPUT=$2

for SEED in 1
do
    python train.py --config-file configs/${CFG}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}

    python inference_eval_text.py --config-file configs/${CFG}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}
done