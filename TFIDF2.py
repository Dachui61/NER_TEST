import xml.etree.ElementTree as ET

# 创建根元素
data = ET.Element('data')

# 添加子元素
id_elem = ET.SubElement(data, 'id')
id_elem.text = '66666'

course_name_elem = ET.SubElement(data, 'courseName')
course_name_elem.text = 'C/C++程序设计'

nodes_elem = ET.SubElement(data, 'nodes')

# 添加节点元素
nodes_data = [
    {'id': 'n1', 'name': '指针', 'type': 'C++基础语法',
     'description': '用于存储变量地址的数据类型，是C++编程中非常重要的概念之一'},
    {'id': 'n2', 'name': '链表', 'type': '数据结构',
     'description': '一种常见的动态数据结构，由多个节点按照一定顺序组成'},
    {'id': 'n3', 'name': '排序算法', 'type': '算法',
     'description': '对一组数据按照一定规则进行排序的算法，包括冒泡排序、选择排序、快速排序等'},
    {'id': 'n4', 'name': '查找算法', 'type': '算法',
     'description': '在一组数据中查找目标元素的算法，包括线性查找、二分查找、哈希查找等'},
    {'id': 'n5', 'name': '栈', 'type': '数据结构',
     'description': '一种遵循后进先出（LIFO）原则的数据结构，常用于程序调试、中断处理等场景'},
    {'id': 'n6', 'name': '队列', 'type': '数据结构',
     'description': '一种遵循先进先出（FIFO）原则的数据结构，常用于任务调度、消息传递等场景'},
    {'id': 'n7', 'name': '二叉树', 'type': '数据结构',
     'description': '一种由节点组成的树形结构，每个节点最多有两个子节点，包括平衡二叉树、红黑树等'},
    {'id': 'n8', 'name': '动态规划', 'type': '算法',
     'description': '一种将复杂问题分解为简单子问题并分步求解的优化策略，常用于处理最优化问题和计算机视觉等领域'},
    {'id': 'n9', 'name': '图论', 'type': '算法',
     'description': '研究图和网络的理论和应用，包括最小生成树、最短路径、拓扑排序等'},
    {'id': 'n10', 'name': '面向对象设计', 'type': 'C++面向对象编程',
     'description': '一种基于对象和类的软件开发方法，强调封装、继承、多态等特性，提高代码复用和可维护性'}
]

for node_data in nodes_data:
    node_elem = ET.SubElement(nodes_elem, 'node')
    node_id_elem = ET.SubElement(node_elem, 'id')
    node_id_elem.text = node_data['id']

    node_name_elem = ET.SubElement(node_elem, 'name')
    node_name_elem.text = node_data['name']

    node_type_elem = ET.SubElement(node_elem, 'type')
    node_type_elem.text = node_data['type']

    node_description_elem = ET.SubElement(node_elem, 'description')
    node_description_elem.text = node_data['description']

links_elem = ET.SubElement(data, 'links')

# 添加链接元素
links_data = [
    {'id': 'l1', 'name': '包含', 'relation': '包含了', 'source': 'n3', 'target': 'n1'},
    {'id': 'l2', 'name': '包含', 'relation': '包含了', 'source': 'n4', 'target': 'n1'},
    {'id': 'l3', 'name':'包含', 'relation': '包含了', 'source': 'n2', 'target': 'n6'},
    {'id': 'l4', 'name': '包含于', 'relation': '包含于', 'source': 'n7', 'target': 'n2'},
    {'id': 'l5', 'name': '包含于', 'relation': '包含于', 'source': 'n8', 'target': 'n3'},
    {'id': 'l6', 'name': '包含', 'relation': '包含了', 'source': 'n9', 'target': 'n2'},
    {'id': 'l7', 'name': '包含于', 'relation': '包含于', 'source': 'n10', 'target': 'n2'},
    {'id': 'l8', 'name': '包含', 'relation': '包含了', 'source': 'n3', 'target': 'n4'},
    {'id': 'l9', 'name': '包含', 'relation': '包含了', 'source': 'n2', 'target': 'n5'},
    {'id': 'l10', 'name': '包含于', 'relation': '包含于', 'source': 'n7', 'target': 'n2'}
]

for link_data in links_data:
    link_elem = ET.SubElement(links_elem, 'link')
    link_id_elem = ET.SubElement(link_elem, 'id')
    link_id_elem.text = link_data['id']
    link_name_elem = ET.SubElement(link_elem, 'name')
    link_name_elem.text = link_data['name']

    link_relation_elem = ET.SubElement(link_elem, 'relation')
    link_relation_elem.text = link_data['relation']

    link_source_elem = ET.SubElement(link_elem, 'source')
    link_source_elem.text = link_data['source']

    link_target_elem = ET.SubElement(link_elem, 'target')
    link_target_elem.text = link_data['target']

# 将数据写入文件
tree = ET.ElementTree(data)
tree.write('data.xml', xml_declaration=True, encoding="utf-8")

xml_string = ET.tostring(data,encoding="utf-8")
xml_string = xml_string.decode('utf-8')
print(xml_string)