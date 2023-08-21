import json


test_file = 'test.json'
result_file = 'gcn_result_test_qtkg_exaqt-tpa.txt'
out_file = r'C:\Users\Myhome\PycharmProjects\EXAQT\analysis\best_xg\gcn_result_test_qtkg_exaqt_tpa_types.txt'

#result_file = r'C:\Users\Myhome\PycharmProjects\EXAQT\baseline\graftnet\analysis\configAnswer_v5c_multiseed_twohop_100_25test.txt'
#out_file = r'C:\Users\Myhome\PycharmProjects\EXAQT\baseline\graftnet\analysis\configAnswer_v5c_multiseed_twohop_100_25_types.txt'

fo = open(out_file, 'w', encoding='utf-8')
result = open(result_file,'r',encoding='utf-8')
res_lines = result.readlines()

with open(test_file, encoding='utf-8') as json_data:
    list = json.load(json_data)
    json_data.close()
type_dic = {'Explicit': [], 'Implicit': [],  'Temp.Ans': [],'Ordinal': []}
p1_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}
h5_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}
mrr_res_dic = {'Explicit': [], 'Implicit': [], 'Temp.Ans': [], 'Ordinal': []}

result_number = {}

count = 0
for item in list:
    source = item["Data source"]
    types = item["Type"]
    id = str(item["Id"])
    for type in types:
        #if type not in type_dic:
        #    type_dic[type] = []
        type_dic[type].append(id)
    count += 1

for line in res_lines:
    if '|' in line and len(line.split('|')) > 3:
        #print (line)
        id = line.split('|')[0]
        p1 = float(line.split('|')[1])
        h5 = float(line.split('|')[2])
        mrr = float(line.split('|')[3])

        for key, value in type_dic.items():
            #print (key)
            #print (value)
            if id in value:
                #print (key)
                #print (id)
                p1_res_dic[key].append(p1)
                h5_res_dic[key].append(h5)
                mrr_res_dic[key].append(mrr)

for key in p1_res_dic.keys():
    print (key)
    if key not in result_number:
        result_number[key] = {}
    result_number[key]['p1'] = str(round(sum(p1_res_dic[key]) / len(p1_res_dic[key]),3))
    print('Average  hits1: ', str(sum(p1_res_dic[key]) / len(p1_res_dic[key])))

for key in mrr_res_dic.keys():
    print(key)
    result_number[key]['mrr'] = str(round(sum(mrr_res_dic[key]) / len(mrr_res_dic[key]), 3))

    print('Average  mrr: ', str(sum(mrr_res_dic[key]) / len(mrr_res_dic[key])))

for key in h5_res_dic.keys():
    print(key)
    result_number[key]['hits5'] = str(round(sum(h5_res_dic[key]) / len(h5_res_dic[key]), 3))

    print('Average  hits5: ', str(sum(h5_res_dic[key]) / len(h5_res_dic[key])))

for key in result_number:
    fo.write(key + '|')
    for item in result_number[key]:
        fo.write(result_number[key][item])
        fo.write('|')
fo.write('\n')
fo.close()

