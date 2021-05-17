lexicons = {
    '\u3000': ' ',
    '一批发鲜鸡': '批发鲜鸡',
    '福生源鞋店': '富生源鞋店',
    '天府理财之“增富理财产品': '天府理财之“增富”理财产品',
}

with open('data/bd-cn-char/trainval/LabelTrain.txt') as f:
    lines = [x.rstrip('\n') for x in f.readlines()]


for old, new in lexicons.items():
    for i, line in enumerate(lines):
        if old in line:
            print(lines[i])
            lines[i] = line.replace(old, new)
            print(lines[i])
            # import pdb; pdb.set_trace()
