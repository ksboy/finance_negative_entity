allennlp train \
    experiments/cls_entity/ensemble/cls_entity_1.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_del_entity_segment/cls_entity_model_1  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_entity/ensemble/cls_entity_2.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_del_entity_segment/cls_entity_model_2  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_entity/ensemble/cls_entity_3.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_del_entity_segment/cls_entity_model_3  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_entity/ensemble/cls_entity_4.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_del_entity_segment/cls_entity_model_4  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/cls_entity/ensemble/cls_entity_5.json  \
    -s ~/workspace/my_models/finance_negative_entity/cls_del_entity_segment/cls_entity_model_5  \
    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_6.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1021_6  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_7.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1021_7  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_8.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1021_8  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_9.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1021_9  \
#    --include-package finance_negative_entity -f
#
#allennlp train \
#    experiments/cls_entity/ensemble/cls_entity_10.json  \
#    -s ~/workspace/my_models/finance_negative_entity/cls_entity/cls_entity_model_1021_10  \
#    --include-package finance_negative_entity -f



