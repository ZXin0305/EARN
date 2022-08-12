
BOX_COOR = [[0,0], [0,1], [0,2], [0,3],
            [1,0], [1,1], [1,2], [1,3],
            [2,0], [2,1], [2,2], [2,3],
            [3,0], [3,1], [3,2], [3,3]
           ]
LABEL = ['walk', 'stand', 'ready', 'next']

def judge_action_label(pre_num):
    """
    根据得到的16数值中最大的数值的序号
    找到对应于输出图中的位置
    从而可以判断里面的两个数值是否相等
    若相等，输出对应的action label
    若不相等，则返回'None',表示当前动作未识别出来
    """
    current_coor = BOX_COOR[int(pre_num)]
    if current_coor[0] != current_coor[1]:
        return 'None'
    elif current_coor[0] == current_coor[1]:
        return LABEL[current_coor[0]]
    
