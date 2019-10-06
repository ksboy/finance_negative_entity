

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

if __name__ == '__main__':
    compare_result('../data/results/result_sentence_entity.csv', '../data/results/result.csv')