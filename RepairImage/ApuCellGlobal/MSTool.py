def toolGetDelimiterIndexList(string:str, delimiter:str)->list:
    list_idx = []
    for idx, char in enumerate(string):
        if char == delimiter:
            list_idx.append(idx)
    return list_idx
