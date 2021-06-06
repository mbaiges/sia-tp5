import json

def bitfield(n):
    return [n >> i & 1 for i in range(7,-1,-1)]

with open('data/fonts.json') as f:
    data = json.load(f)
    new_data = {}

    for k in data:
        v = data[k]

        arr1 = []
        for char_idx in range(0, len(v)):
            img_arr = v[char_idx]

            arr2 = []
            for n in img_arr:
                arr2.append(bitfield(n)[3:])

            arr1.append(arr2)

        new_data[k] = arr1

    s = ''

    for k in new_data:
        
        s += k + ':\n'
        letter_list = new_data[k]

        for letter in letter_list:
            s += '  letter\n'

            for row in letter:
                s += '    ' + str(row) + '\n'

            s += '\n'
        
        s += '\n\n'

    print(s)

    with open('data/fonts_training.json', 'w') as json_file:
        json.dump(new_data, json_file)