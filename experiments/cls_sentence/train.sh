allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_1  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_2  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_3.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_3  \
    --include-package finance_negative_entity -f
    
allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_4.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_4  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_5.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_model_5  \
    --include-package finance_negative_entity -f