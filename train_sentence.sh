allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_1  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_2  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_3.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_3  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_4.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_4  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_5.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_5  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_6.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_6  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_7.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_7  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_8.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_8  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_9.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_9  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_sentence/ensemble/cls_sentence_10.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_sentence/cls_sentence_roberta_relu_1024_10  \
    --include-package finance_negative_entity -f

#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_5.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_5  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_6.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_6  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_7.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_7  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_8.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_8  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_9.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_9  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_10.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1536_1016_10  \
#    --include-package finance_negative_entity -f
