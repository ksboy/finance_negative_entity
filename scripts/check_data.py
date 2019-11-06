import csv
import json


def check_train_file(input_file):
    key_entity_dict = {}
    entity_dict = {}
    del_entity_list = []
    rows = []
    neg_sentence_count = 0
    neg_entity_count = 0
    entity_count = 0
    del_entity_count = 0
    count = 0
    del_sentence_count = 0
    entity_count = 0
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            # if row['negative'] == "1":
            #     neg_sentence_count += 1
            entity_list = row['entity'].split(";")
            entity_count += len(entity_list)
            # key_entity_list = row['key_entity'].split(";")
            # neg_entity_count += len(key_entity_list)
            for entity in entity_list:
                entity_count += 1
                if entity_dict.get(entity,-1) != -1:
                    entity_dict[entity] += 1
                else:
                    entity_dict[entity] = 1
            # for key_entity in key_entity_list:
            #     if key_entity_dict.get(key_entity,-1) != -1:
            #         key_entity_dict[key_entity] += 1
            #     else:
            #         key_entity_dict[key_entity] = 1
        del entity_dict['']
        # del key_entity_dict['']
        for entity in key_entity_dict.keys():
            if key_entity_dict[entity]>3 and key_entity_dict[entity] > entity_dict[entity] * 0.7:
                del_entity_list.append(entity)
                del_entity_count += entity_dict[entity]

        for entity in del_entity_list:
            print(entity, entity_dict[entity], key_entity_dict[entity])

        with open(input_file, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            title = reader.fieldnames
            for row in reader:
                for del_entity in del_entity_list:
                    if del_entity in row['entity']:
                        del_sentence_count +=1
                        break

        key_entity_dict = sorted(key_entity_dict.items(), key=lambda x: x[1], reverse=True)
        entity_dict = sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)

        print("entity_list:"+str(entity_dict))
        print("key_entity_list:"+str(key_entity_dict))
        print("del_entity_list:", del_entity_list)

        print(count)
        print("neg_sentence:", neg_sentence_count)
        print("entity_count:", entity_count)
        print("neg_entity_count:", neg_entity_count)
        print("del_sentencce_count:", del_sentence_count)
        print("entity_count:", entity_count)
        print("del_entity_count:", del_entity_count)

        return del_entity_list


def check_label(input_file):
    positive_count = 0
    negative_count = 0
    with open(input_file, encoding='utf-8') as infile:
        items = []
        for row in infile:
            row = json.loads(row)
            if row['label'] == "正类":
                positive_count += 1
            else:
                negative_count += 1
    print("Positive count:", positive_count)
    print("Negative_count:", negative_count)


def check_test_file(input_file):
    entity_dict = {}
    rows = []
    large_count = 0
    small_count = 0
    count = 0
    count1 = 0
    count2 = 0
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            entity_list = row['entity'].split(";")
            for entity in entity_list:
                if entity_dict.get(entity,-1) != -1:
                    entity_dict[entity] += 1
                else:
                    entity_dict[entity] = 1
        del entity_dict['']

        entity_dict = sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)

        print("entity_list:"+str(entity_dict))


        print(large_count)
        print(small_count)
        print(count)
        print(count1)


def check_dev_data(input_file):
    entity_dict = {}
    count = 0
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            entity = row['question']
            if entity_dict.get(entity, -1) != -1:
                entity_dict[entity] += 1
            else:
                entity_dict[entity] = 1

        entity_dict = sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)

    del_entity_list = check_train_file(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/Train_Data.csv")
    for entity in entity_dict:
        if entity not in del_entity_list:
            count += 1

    print(entity_dict)
    print(count)


if __name__ == '__main__':
    # check_train_file(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/Round2_train.csv")
    check_train_file(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/round2_test.csv")
    # check_label(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/ensemble_data/cls_entity_1030/train_1.jsonl")
    # check_label(
    #     input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/ensemble_data/cls_entity_1030/dev_1.jsonl")
    # check_dev_data(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/results/dev_test.csv")