cd evaluation/passkey

MODEL=Yarn-Llama-2-7b-128k
MODELPATH=NousResearch/Yarn-Llama-2-7b-128k
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=100000

for token_budget in 256 512 1024
do
    python passkey.py -m $MODELPATH \
        --iterations 50 --fixed-length $length \
        --quest --token_budget $token_budget --chunk_size 16 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
done
