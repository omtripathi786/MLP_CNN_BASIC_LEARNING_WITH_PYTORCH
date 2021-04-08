def get_least_occ_data(list_arg) :
    dict_data = {}
    for i in list_arg :
        if not i in dict_data :
            dict_data[i] = 0

        dict_data[i] += 1
    least_occ = 0
    least_occ_key = ""
    for i in dict_data :
        if least_occ > dict_data[i] :
            least_occ = dict_data[i]
            least_occ_key = i
    return least_occ_key

if __name__ == '__main__':
    l = [1, 2, 9, 6, 5, 2, 9, 1]
    print(get_least_occ_data([1,1,2,3,3,3,4,4,4,4]))
