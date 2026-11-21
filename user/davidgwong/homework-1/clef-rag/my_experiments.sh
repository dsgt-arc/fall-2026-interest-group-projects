weight_pairs=(
    "0.2 0.8"
    "0.5 0.5"
    "0.8 0.2"
)

candidate_amounts=(5 20 25)
top_ks=(3 6 9)

experiment_start=$(date +%s)
echo "***STARTING EXPERIMENT.*** Start time: $(date)"

for amount in "${candidate_amounts[@]}"; do
    for k in "${top_ks[@]}"; do
        for pair in "${weight_pairs[@]}"; do
            read -r w1 w2 <<< "$pair"
            echo "---"
            echo "Running with candidate amount: $amount"
            echo "Running with top_k: $k"
            echo "Running with weights: $w1, $w2"
            start_time=$(date +%s)
            echo "Start time: $(date)"

            uv run clef-rag evaluate \
                --questions "eval/my_questions.json" \
                --max_candidates "$amount" \
                --top_k "$k" \
                --retriever-weights "$w1" "$w2" \
                --output "results/my_question_results_02/02_eval_d768-c${amount}_-k${k}_w${w1}-${w2}.json"

            end_time=$(date +%s)
            elapsed=$(( end_time - start_time ))
            echo "End time: $(date), Elapsed time: ${elapsed}s"
        done
    done
done

echo "---"
experiment_end=$(date +%s)
total_elapsed=$(( experiment_end - experiment_start ))
echo "***ENDING EXPERIMENT.*** End time: $(date), total elapsed time: ${total_elapsed}s"