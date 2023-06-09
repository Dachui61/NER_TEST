import pdfplumber
import pandas as pd

with pdfplumber.open("./pdf/在Spring Boot上使用Flask.pdf") as pdf:
    print(pdf.metadata)
    # print("总页数："+str(len(pdf.pages))) #总页数
    print("pdf文档总页数:", len(pdf.pages))
# 读取第一页的宽度，页高等信息
    # 第一页pdfplumber.Page实例
    first_page = pdf.pages[0]
    # 查看页码
    print('页码：', first_page.page_number)

    # 查看页宽
    print('页宽：', first_page.width)

    # 查看页高
    print('页高：', first_page.height)

    print(pdf.pages[0].extract_text())

