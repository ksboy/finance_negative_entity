allennlp train \
    /Users/xmh_mac/2019Projects/finance_negative_entity/experiments/ner/ensemble/ner_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_1 \
    --include-package finance_negative_entity -f

allennlp train \
    /Users/xmh_mac/2019Projects/finance_negative_entity/experiments/ner/ensemble/ner_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_2 \
    --include-package finance_negative_entity -f

allennlp train \
    /Users/xmh_mac/2019Projects/finance_negative_entity/experiments/ner/ensemble/ner_3.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_3 \
    --include-package finance_negative_entity -f
    
allennlp train \
    /Users/xmh_mac/2019Projects/finance_negative_entity/experiments/ner/ensemble/ner_4.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_4 \
    --include-package finance_negative_entity -f

allennlp train \
    /Users/xmh_mac/2019Projects/finance_negative_entity/experiments/ner/ensemble/ner_5.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_ner_5 \
    --include-package finance_negative_entity -f