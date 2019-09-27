allennlp predict  \
       ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_withTitle \
       data/processed_data/test.jsonl \
       --output-file data/results/predict.jsonl \
       --batch-size 4 \
       --silent \
       --use-dataset-reader  \
       --include-package finance_negative_entity \
       --predictor bert_seq_pair_clf