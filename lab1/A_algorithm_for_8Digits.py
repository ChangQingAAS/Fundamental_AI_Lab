# -*- Coding: UTF-8 -*-
# BB.py
# @作者 王红阳
# @创建日期 2021-05-21T18:54:40.671Z+08:00
# @最后修改日期 2021-05-21T18:54:46.933Z+08:00
# @代码说明
#


# 计算状态对应的逆序数,确定是奇数列还是偶数列，用于后续判断
# 若不是相同类型，则无解，直接退出即可
def get_odd_even_counter(node):
    odd_even_counter = 0
    for i in range(1, 9):
        for j in range(0, i):
            if node[j] > node[i] and node[i] != '0':
                odd_even_counter += 1
    return odd_even_counter


# 用于计算评估函数中的h
def calculate_h(node):
    global goal_node
    h = 0

    # h(n)为不在位数码个数。
    # for i in range(0, 9):
    #     if node[i] != '0':
    #         if goal_node[i] != node[i]:
    #             h += 1

    # h(n)为错位的牌必须要移动的距离之和
    for i in range(0, 9):
        if node[i] != '0':
            # i为某元素的当前位置，j为该元素在目标状态下的位置
            j = goal_node.index(node[i])
            h += postion_to_position[i][j]
    return h


# 拓展node状态对应的子结点
def expand_child_node(node):
    global states_for_expand
    child_nodes = []
    # 找到0，即空格所在的位置，从而进行转换
    space_postion = node.index("0")
    state_list = states_for_expand[space_postion]
    j = space_postion
    # i 是可以与空格交换的位置
    for i in state_list:
        if i > j:
            i, j = j, i
        # 因为是一维的，所以用这样切分的方式模拟二维层面的转换
        newNode = node[:i] + node[j] + node[i + 1:j] + node[i] + node[j + 1:]
        child_nodes.append(newNode)
        # 千万不要忘记把j复原
        j = space_postion
    return child_nodes


# 输出结果
def print_result_path(result_path):
    print()
    for i in range(len(result_path)):
        print("第 " + str(i) + " 步：")
        for j in range(0, 9):
            if j % 3 == 0:
                if result_path[i][j] != '0':
                    print("    ", result_path[i][j], end=' ')
                else:
                    print("      ", end=' ')
            elif j % 3 == 1:
                if result_path[i][j] != '0':
                    print(result_path[i][j], end=' ')
                else:
                    print(' ', end=' ')
            else:
                if result_path[i][j] != '0':
                    print(result_path[i][j])
                else:
                    print()
        print()


# 选择openlist表中的最小的估价函数值对应的节点
def get_minF_node(opened):
    temp_dict = {}
    for node in openlist:
        k = f[node]
        temp_dict[node] = k
    minF_node = min(temp_dict, key=temp_dict.get)
    # for debugging
    # print("minF_node is:")
    # for j in range(0, 9):
    #     if j % 3 != 2:
    #         print(minF_node[j], end=' ')
    #     else:
    #         print(minF_node[j])
    # print("g is", g[minF_node])
    # print("f is", f[minF_node])
    # print("h is", f[minF_node] - g[minF_node])
    return minF_node


# 用parent导出由初始到目标状态的路径
def find_result_path(current_node):
    result_path.append(current_node)
    # 根据parent字典中存储的父结点提取路径中的结点
    while parent[current_node] != -1:
        current_node = parent[current_node]
        result_path.append(current_node)
    # 逆序
    result_path.reverse()


# 主程序开始
if __name__ == '__main__':

    # open表
    openlist = []
    # close表
    closedlist = []
    # 评估函数中的g(n),即初始结点到当前结点的路径长度
    g = {}
    # 评估函数中的f(n)，即该状态对应的估价函数值
    f = {}
    # 用于存储各个状态对应的父结点
    parent = {}
    # 用来存放路径
    result_path = []

    # expand中存储的是九宫格中每个位置对应的可以移动的情况，
    # 例如：将一维数组中号元素可以映射到3*3数组的中央，即4号元素可以上下左右地移动
    # 当确定0的位置，即空格的位置，就可以进行移动
    states_for_expand = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [3, 1, 5, 7],
        5: [4, 2, 8],
        6: [3, 7],
        7: [6, 4, 8],
        8: [7, 5]
    }
    # positon_to_position 为棋盘中某位置的元素到另一位置的最小距离（假定路径中间没有路障
    postion_to_position = [[0, 1, 2, 1, 2, 3, 2, 3, 4],
                           [1, 0, 1, 2, 1, 2, 3, 2, 3],
                           [2, 1, 0, 3, 2, 1, 4, 3, 2],
                           [1, 2, 3, 0, 1, 2, 1, 2, 3],
                           [2, 1, 2, 1, 0, 1, 2, 1, 2],
                           [3, 2, 1, 2, 1, 0, 3, 2, 1],
                           [2, 3, 4, 1, 2, 3, 0, 1, 2],
                           [3, 2, 3, 2, 1, 4, 1, 0, 1],
                           [4, 3, 2, 3, 2, 1, 2, 1, 0]]
    start_node = input("请输入初始节点(以一维数组形式）：")
    goal_node = input("请输入目标节点：")

    if start_node == goal_node:
        print("初始状态和目标状态一致！")
    # 判断从初始状态是否可以达到目标状态
    if (get_odd_even_counter(start_node) %
            2) != (get_odd_even_counter(goal_node) % 2):
        print("该目标状态不可达,无解！")
    else:
        # 将初始结点存入opened表

        # 令初始节点的父节点为-1，方便后面寻路
        parent[start_node] = -1
        # 初始节点的g(n)为0（根节点层数为0）
        g[start_node] = 0
        # 计算初始结点的f,没有保存h是因为它的作用就是更新f。
        f[start_node] = g[start_node] + calculate_h(start_node)
        openlist.append(start_node)

        while openlist:
            # 选择估价函数f值最小的节点
            # print("entered get_minF_node()")
            current_node = get_minF_node(openlist)
            del f[current_node]
            # 将要遍历的结点从open表中删除
            openlist.remove(current_node)

            # 如果当前节点是目标状态，则退出
            if current_node == goal_node:
                break

            # 当前节点不在closed表
            if current_node not in closedlist:

                # 将当前节点存入closed表
                closedlist.append(current_node)

                # 扩展子结点
                child_nodes = expand_child_node(current_node)

                for node in child_nodes:
                    # 如果子结点在open和close表中都未出现，则存入open表
                    # 并求出对应的估价函数值
                    if node not in openlist and node not in closedlist:
                        g[node] = g[current_node] + 1
                        f[node] = g[node] + calculate_h(node)
                        parent[node] = current_node
                        openlist.append(node)
                    else:
                        # 若子结点已经在open表中，则尝试更新其估值函数f值
                        # 同时改变parent字典
                        if node in openlist:
                            fn = g[current_node] + 1 + calculate_h(node)
                            if fn < f[node]:
                                f[node] = fn
                                parent[node] = current_node

        # 找从起始点到终点的路径
        find_result_path(current_node)
        # 按格式输出结果
        print_result_path(result_path)
