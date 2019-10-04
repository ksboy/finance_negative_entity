allennlp train \
    experiments/cls_entity/ensemble/cls_entity_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_BERT_ReLU_temp1 \
    --include-package finance_negative_entity -f

#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_2.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_BERT_ReLU_2 \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_3.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_BERT_ReLU_3 \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_4.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_BERT_ReLU_4 \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_5.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_BERT_ReLU_5 \
#    --include-package finance_negative_entity -f