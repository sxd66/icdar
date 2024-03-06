import pandas as pd
def find_enclosed_substrings(long_string):
    # 初始化结果列表
    enclosed_substrings = []

    # 遍历字符串，查找包围的子串
    for i in range(len(long_string)):
        if long_string[i:i + 5] == ".png " :
            # 找到以“//”开头的子串
            start_index = i+5
            for j in range(i + 5, len(long_string)):
                if long_string[j:j + 1] == "\n" :
                    # 找到以“//”结尾的子串
                    end_index = j
                    enclosed_substrings.append(long_string[start_index:end_index])
                    break
    enclosed_substrings.append('Parks')
    return enclosed_substrings
file_path='../data/train_labels.txt'
with open(file_path,'r') as file:
    fp=file.read()


list=find_enclosed_substrings(fp)
diction={'label':list}

df=pd.DataFrame(diction)
df.to_csv('data.csv',index=False)


