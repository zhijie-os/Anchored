cd evaluation/passkey

MODEL=longchat-7b-v1.5-32k
MODELPATH=lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100

for token_budget in 128
do
    python passkey.py -m $MODELPATH \
        --iterations 50 --fixed-length $length \
        --quest --token_budget $token_budget --chunk_size 16 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
done
