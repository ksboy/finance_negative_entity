allennlp train \
    experiments/cls_entity/ensemble/cls_entity_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_2 \
    --include-package finance_negative_entity -f