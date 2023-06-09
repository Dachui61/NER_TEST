import re

from paddlenlp import Taskflow

# schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
# ie = Taskflow('information_extraction', schema=schema , home_path= "/paddle")
# print(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))
for item in ["第一章", "第二章", "第三章", "第四章", "第五章"]:
    with open("./data/raw/" + item + ".txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        # 去掉句首的空格
        lines = [line.strip() for line in lines]
        # title = []  # 保留标题
        for line in lines:
            if line == "":
                continue
            if line.startswith('#'):  # 读取到了标题行
                # if title != '':
                    print(line)
                # pattern = r'#\d+\.\s+(.*)'
                # title = re.findall(r'#\d+[\.\d+]* (\w+)', line)[0]  # 提取出标题
                    title = re.findall(r'#\d+\.\d+ (\w+)', line)  # 提取出二级标题
                    if len(title) != 0:
                        print(title[0])
