import csv
import json


def compare_result(input_file_1, input_file_2):
    data_1=[]
    with open(input_file_1, encoding='utf-8') as infile:
        for row in infile:
            data_1.append(row)
    
    data_2=[]
    with open(input_file_2, encoding='utf-8') as infile:
        for row in infile:
            data_2.append(row)
    
    print("len(set(data_1))",len(set(data_1)))
    print("len(set(data_2))",len(set(data_2)))
    print("len(set(data_1)&set(data_2))", len(set(data_1)&set(data_2)))
    print("len(set(data_1)-set(data_2))", len(set(data_1)-set(data_2)))
    print("len(set(data_2)-set(data_1))", len(set(data_2)-set(data_1)))


def gen_large_result_entity(input_file, output_file, test_file):
    rows = []
    raw_rows = []
    count = 0
    empty_ids = []
    empty_count = 0
    with open(test_file, encoding='utf-8') as test_file:
        for row in test_file:
            row = json.loads(row)
            if row['passage'] == "":
                empty_ids.append(row['id'])
            raw_rows.append(row)
    print(empty_ids)

    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            if row['id'] in empty_ids:
                if row['negative'] == "1":
                    row['negative'] = "0"
                    row['key_entity'] = ""
                    empty_count += 1
            if ";" not in row["key_entity"]:
                rows.append(row)
                continue
            key_entity_list = row["key_entity"].split(";")
            for currentity in key_entity_list:
                for entity in key_entity_list:
                    if currentity in entity and currentity != entity:
                        if currentity in key_entity_list:
                            key_entity_list.remove(currentity)
                            print(row)
                            count += 1
            
            row["key_entity"] = ";".join(key_entity_list)
            rows.append(row)
        print(count)
        print("empty_count", empty_count)
    with open(output_file, 'w+') as output:
        writer = csv.DictWriter(output, fieldnames=title)
        for row in rows:
            writer.writerow(row)


def gen_single_negative(input_file, output_file):
    rows = []
    with open(input_file, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        title = reader.fieldnames
        for row in reader:
            row['key_entity'] = ""
            rows.append(row)
    with open(output_file, 'w+') as output:
        writer = csv.DictWriter(output, fieldnames=title)
        for row in rows:
            writer.writerow(row)


def check_train_file(input_file):
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
            key_entity_list = row['key_entity'].split(";")
            for entity in entity_list:
                if "????" in entity:
                    print(row['entity'], row['key_entity'])
                    count += 1

            # for entity in entity_list:
            #     if "????" in entity:
            #         print(row['\ufeffid'], entity_list)
            #         count += 1
            #         for curr in entity_list:
            #             if curr in entity and curr != entity:
            #
            #                 count1 += 1


        print(large_count)
        print(small_count)
        print(count)
        print(count1)


def check_test_file(input_file, output_file):
    with open(input_file, encoding='utf-8') as infile:
        items = []
        count = 0
        empty_passage = 0
        empty_entity = 0
        dup_entity = 0
        for row in infile:
            row = json.loads(row)
            item = {}
            item['id'] = row['id']
            if not row['passage']:
                empty_passage += 1
                print(row)
                continue
            item['passage'] = row['passage']
            item['entity'] = []
            for entity in row['entity']:
                if entity not in item['passage']:
                    # print(item['id'], item['passage'],entity)
                    count += 1
                    continue
                item['entity'].append(entity)
            if not item['entity']:
                # print(item['passage'], row['entity'])
                empty_entity += 1
            # for curr in item['entity']:
            #     for entity in item['entity']:
            #         if curr in entity and curr != entity:
            #             dup_entity += 1
            #             print(item['id'], item['entity'])
            items.append(item)
    # with open(output_file, 'w+') as outfile:
    #     for item in items:
    #         json.dump(item, outfile)

    print("Entity not in text", count)
    print("Empty passage", empty_passage)
    print("Empty entities", empty_entity)
    print("Duplicate entities", dup_entity)


if __name__ == '__main__':

    # compare_result('../data/results/result_sentence_entity.csv', '../data/results/result.csv')

    gen_large_result_entity(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/results/result_test_cleansed1_roberta_logits6_1024.csv",
                            output_file="/home/minghongxia/BDCI/finance_negative_entity/data/results/result_test_cleansed2_roberta_logits6_1024.csv",
                            test_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/test_cleansed_1015.jsonl")

    # check_test_file(input_file="/home/mhxia/BDCI/finance_negative_entity/data/processed_data/test.jsonl",
    #                 output_file="/home/mhxia/BDCI/finance_negative_entity/data/processed_data/test_cleansed.jsonl")

    # check_train_file(input_file="/home/mhxia/BDCI/finance_negative_entity/data/processed_data/Train_Data.csv")
