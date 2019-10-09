allennlp train \
    experiments/ner/ensemble/ner_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_6 \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_7 \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_3.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_8 \
    --include-package finance_negative_entity -f
    
allennlp train \
    experiments/ner/ensemble/ner_4.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_9 \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_5.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_10 \
    --include-package finance_negative_entity -f