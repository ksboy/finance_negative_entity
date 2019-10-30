import csv


def check_train_file(input_file):
    key_entity_dict = {}
    entity_dict = {}
    del_entity_list = []
    rows = []
    del_count = 0
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
                count2 += 1
                if entity_dict.get(entity,-1) != -1:
                    entity_dict[entity] += 1
                else:
                    entity_dict[entity] = 1
            for key_entity in key_entity_list:
                if key_entity_dict.get(key_entity,-1) != -1:
                    key_entity_dict[key_entity] += 1
                else:
                    key_entity_dict[key_entity] = 1

        del entity_dict['']
        del key_entity_dict['']

        for entity in key_entity_dict.keys():
            if key_entity_dict[entity]>3 and key_entity_dict[entity] > entity_dict[entity] * 0.7:
                del_entity_list.append(entity)
                del_count += entity_dict[entity]

        for entity in del_entity_list:
            print(entity, entity_dict[entity], key_entity_dict[entity])

        with open(input_file, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            title = reader.fieldnames
            for row in reader:
                for del_entity in del_entity_list:
                    if del_entity in row['entity']:
                        count1 +=1
                        break

        key_entity_dict = sorted(key_entity_dict.items(), key=lambda x: x[1], reverse=True)
        entity_dict = sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)

        print("entity_list:"+str(entity_dict))
        print("key_entity_list:"+str(key_entity_dict))
        print(del_entity_list)




        print(count)
        print(count1)
        print(count2)
        print(del_count)


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


if __name__ == '__main__':
    check_train_file(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/Train_Data.csv")
    check_test_file(input_file="/home/minghongxia/BDCI/finance_negative_entity/data/processed_data/Test_Data.csv")