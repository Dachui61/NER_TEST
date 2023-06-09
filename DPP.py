# from paddlenlp import Taskflow
# ddp = Taskflow("dependency_parsing")
# res = ddp(["2月8日谷爱凌夺得北京冬奥会第三金", "他送了一本书"])

import requests

# 循环请求288个图片链接
for i in range(1, 289):
    requests.packages.urllib3.disable_warnings()  # 禁用警告和错误消息
    url = f'https://www.cmsy.fun/medias/randomGallery/7%E9%9A%8F%E6%9C%BA{i+10000}.jpg'
    response = requests.get(url, verify=False)

    # 下载图片到本地文件夹
    if response.status_code == 200:
        with open(f'./images/{i}.jpg', 'wb') as f:
            f.write(response.content)
