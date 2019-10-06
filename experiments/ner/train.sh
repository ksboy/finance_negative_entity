allennlp train \
    experiments/ner/ensemble/ner_1.json \
    -s ~/workspace/my_models/finance_negative_entity/ner_neg/ner_neg_model_1  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_2.json \
    -s ~/workspace/my_models/finance_negative_entity/ner_neg/ner_neg_model_2  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_3.json \
    -s ~/workspace/my_models/finance_negative_entity/ner_neg/ner_neg_model_3  \
    --include-package finance_negative_entity -f
    
allennlp train \
    experiments/ner/ensemble/ner_4.json \
    -s ~/workspace/my_models/finance_negative_entity/ner_neg/ner_neg_model_4  \
    --include-package finance_negative_entity -f

allennlp train \
    experiments/ner/ensemble/ner_5.json \
    -s ~/workspace/my_models/finance_negative_entity/ner_neg/ner_neg_model_5  \
    --include-package finance_negative_entity -f