def seq_array(arr, seq):
    seq_id = 0
    for number in arr:
        if seq_id == len(seq):
            break
        if seq[seq_id] == number:
            seq_id += 1
    return seq_id == len(seq)


if __name__ == '__main__':
    l = [1, 2, 3, 4]
    sub_l = [1, 4]
    print(seq_array(l, sub_l))
