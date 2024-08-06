# encoding=utf-8
import math
import numpy as np

# 计算两点间距离
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# 计算三点形成的一个角度
def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180
    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return abs(angle)

# 计算边界框的 长宽比/宽高比
# [x_min, y_min, x_max, y_max]
def aspectRatio(box):
    box_list = box.tolist()
    h = box_list[3]-box_list[1]
    w = box_list[2]-box_list[0]
    bi = w / h
    return [bi, w, h]

# 判断是否摔倒
def is_fallen(keypoints, boxes):
    keypoints = np.array(keypoints)

    # 定义一些用于判断姿态的阈值
    # 未被检测到的关键点数量
    ATHERPOSE = 0
    # 人体中心线与水平线的最大允许夹角 90为直立
    HUMAN_ANGLE = 75
    # 检测区域的最大宽高比3/5
    ASPECT_RATIO = 0.6
    # 用于判断关键点在垂直和水平方向上的差异
    Vertical_threshold = 0
    Horizontal_threshold = 0

    # 检查关键点是否有被检测到
    # 对于每个关键点，如果它的 x （0）和 y（1） 坐标之和为 0，则表示该关键点没有被检测到
    # 在这种情况下，ATHERPOSE 变量将增加 1
    # 这种检查方式基于一个假设：如果关键点未被检测到，则其坐标被设置为 (0, 0)
    Left_Shoulder = keypoints[5]
    # if Left_Shoulder[0] + Left_Shoulder[1] == 0:
    #     ATHERPOSE += 1
    Right_Shoulder = keypoints[6]
    # if Right_Shoulder[0] + Right_Shoulder[1] == 0:
    #     ATHERPOSE += 1
    Left_Hip = keypoints[11]
    # if Left_Hip[0] + Left_Hip[1] == 0:
    #     ATHERPOSE += 1
    Right_Hip = keypoints[12]
    # if Right_Hip[0] + Right_Hip[1] == 0:
    #     ATHERPOSE += 1
    Left_Knee = keypoints[13]
    # if Left_Knee[0] + Left_Knee[1] == 0:
    #     ATHERPOSE += 1
    Right_Knee = keypoints[14]
    # if Right_Knee[0] + Right_Knee[1] == 0:
    #     ATHERPOSE += 1
    Left_Ankle = keypoints[15]
    # if Left_Ankle[0] + Left_Ankle[1] == 0:
    #     ATHERPOSE += 1
    Right_Ankle = keypoints[16]
    # if Right_Ankle[0] + Right_Ankle[1] == 0:
    #     ATHERPOSE += 1



    # 检查关键点是否有被检测到，如果关键点的 x 和 y 坐标之和为 0，则表示该关键点没有被检测到
    keypoint_indices = [5, 6, 11, 12, 13, 14, 15, 16]  # 肩、髋、膝、踝的关键点索引
    for index in keypoint_indices:
        if keypoints[index][0] + keypoints[index][1] == 0:
            ATHERPOSE += 1



    # 计算左右肩膀的中心点
    Shoulders_c = np.array([(Left_Shoulder[0] + Right_Shoulder[0]) // 2,
                            (Left_Shoulder[1] + Right_Shoulder[1]) // 2])
    # 计算左右胯部的中心点
    hips_c = np.array([(Left_Hip[0] + Right_Hip[0]) // 2,
              (Left_Hip[1] + Right_Hip[1]) // 2])
    # 计算左右膝盖的中心点
    Knee_c = np.array([(Left_Knee[0] + Right_Knee[0]) // 2,
              (Left_Knee[1] + Right_Knee[1]) // 2])
    # 计算左右脚踝的中心点
    Ankle_c = np.array([(Left_Ankle[0] + Right_Ankle[0]) // 2,
               (Left_Ankle[1] + Right_Ankle[1]) // 2])
    # 计算身体中心线与水平线夹角 检测弯腰等
    # np.array([0, hips_c[1]])创建一个新的点，这个点位于水平线上，与髋关节中心点的y坐标相同，x坐标设为0
    # 这个点的目的是作为水平参考线的一部分，帮助计算与水平线的夹角
    human_angle = get_angle(Shoulders_c, hips_c, np.array([0, hips_c[1]]))
    # 计算检测区域宽高比 
    aspect_ratio = aspectRatio(boxes)
    # print("宽高比::", aspect_ratio)
    # 计算肩部中心点与胯部中心点的垂直距离差
    #human_shoulderhip = abs(Shoulders_c[1] - hips_c[1])

    # 计算肩部胯部膝盖夹角
    Hip_Knee_Shoulders_angle = get_angle(Shoulders_c, hips_c, Knee_c)
    # Hip_Knee_Right_angle = get_angle(Right_Shoulder, Right_Hip, Right_Knee)
    # print(Hip_Knee_Shoulders_angle, Hip_Knee_Right_angle)

    # 计算胯部膝盖小腿夹角
    Ankle_Knee_Hip_angle = get_angle(hips_c, Knee_c, Ankle_c)
    # Ankle_Knee_Right_angle = get_angle(Right_Hip, Right_Knee, Right_Ankle)

    # 计算膝盖与肩膀是否处于相似的垂直位置（左膝和左肩在垂直方向上的相对位置
    # 判断蹲下、弯腰
    # 如在举重或体操中保持正确的身体对齐，摔倒时垂直距离接近为0
    # 图像坐标系通常定义为左上角为原点(0,0)，向下和向右为正方向，所以是膝盖减肩膀=正
    vertical_threshold = (Left_Knee[1] - Left_Shoulder[1])/aspect_ratio[-1]

    # 计算膝盖与肩膀是否处于相似的水平位置（左肩和左膝在水平方向上的相对位置
    # 判断侧向移动，摔倒时肩膀膝盖水平距离拉长
    horizontal_threshold = (Left_Shoulder[0] - Left_Knee[0]) / aspect_ratio[1]

    # 初始化站、坐、摔倒的指数值
    # _weight 变量用于记录调整评分的原因和数值，便于调试和理解评分的调整原因。
    status_score = {'Stand': 0.0,
                    'Fall': 0.0,
                    'Sit': 0.0,
                    'other': 0.0}
    _weight = ''

    #图像的质量可能不足或者被检测对象的某些部分被遮挡了
    if ATHERPOSE >= 8:
        status_score['other'] += 5.6

    # 判断膝盖和髋部的中心点是否同时未被检测到
    # 如果这些关键点未被检测到，则减少Fall和Stand的得分，而增加Sit的得分
    if Knee_c[0] == 0 and Knee_c[1] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += 0.6
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += -0.8 * 2
        _weight = f'[1]Sit:+0.6, Fall:-1.6 ,Stand: -1.6'

    # 判断肩部和髋部的中心点是否同时未被检测到
    # 如果这些关键点未被检测到，则减少Sit和Fall的得分，而增加Stand的得分
    elif Shoulders_c[1] == 0 and Shoulders_c[0] == 0 and hips_c[0] == 0 and hips_c[1] == 0:
        status_score['Sit'] += -0.8 * 2
        status_score['Fall'] += -0.8 * 2
        status_score['Stand'] += 0.6
        _weight = f'[1]Sit:-1.6, Fall:-1.6 ,Stand: +0.6'
    
    # 胯部、膝盖与小腿形成的角度
    else:
        # 如果角度在125到180之间，并且宽高比小于阈值（ASPECT_RATIO），
        # 增加Stand的得分
        # 这种角度和宽高比组合可能表示身体较直，符合站立的姿态 
        if 180 > Ankle_Knee_Hip_angle > 165 and aspect_ratio[0] < ASPECT_RATIO:
            status_score['Stand'] += 1.6
            _weight = f'[1]Stand:+1.6'

        # 如果角度在75以下，并且高度小于宽度
        # 则增加Fall的得分
        # 这种情况可能代表身体水平于地面
        if 75 > Ankle_Knee_Hip_angle and aspect_ratio[0] > 1 / ASPECT_RATIO:
            status_score['Fall'] += 1.6
            _weight = f'[1]Fall:+1.6'

        # 如果角度在75到125之间，
        # 增加Sit的得分
        # 这可能代表身体是半坐半躺的状态
        if 165 > Ankle_Knee_Hip_angle > 75:
            status_score['Sit'] += 0.6
            _weight = f'[1]Sit:+0.6'



    # 身体中心线与水平线夹角+-75，相对直立或稍微倾斜
    if human_angle in range(-HUMAN_ANGLE, HUMAN_ANGLE):
        status_score['Stand'] += 1.0
        status_score['Fall'] += 0.2
        status_score['Sit'] += 0.1
        _weight = f'{_weight}, [2]Stand:+1.0, Fall:+0.2, Sit:+0.1'
    else:
        # Fall的得分取决于与90度的差值，角度越小，跌倒的概率越大
        status_score['Fall'] += 0.8 * ((90 - human_angle) / 90)
        _weight = f'{_weight}, [3]Fall:+{0.8 * ((90 - human_angle) / 90)}'

    # 宽高比小于0.6且human angle在65-115则为站立
    if aspect_ratio[0] < ASPECT_RATIO and human_angle in range(65, 115):
        status_score['Stand'] += 0.8
        _weight = f'{_weight}, [4]Stand:+0.8'

    elif aspect_ratio[0] > 1 / ASPECT_RATIO:  # 5/3 宽高比异常
        status_score['Fall'] += 0.8
        _weight = f'{_weight}, [5]Fall:+0.8'



# 垂直和水平的参数低于期望值
    if vertical_threshold < Vertical_threshold:
        status_score['Fall'] += 0.6
        status_score['Sit'] += -0.15
        _weight = f'{_weight}, [5]Fall:+0.6, Sit:-0.15'
    if horizontal_threshold < Horizontal_threshold:
        status_score['Fall'] += 0.6
        status_score['Sit'] += -0.15
        _weight = f'{_weight}, [5]Fall:+0.6, Sit:-0.15'


    def update_scores(status_score, sit_increment, stand_increment, fall_increment):
        if sit_increment:
            status_score['Sit'] += sit_increment
        if stand_increment:
            status_score['Stand'] += stand_increment
        if fall_increment:
            status_score['Fall'] += fall_increment

    def process_angles(angle1, angle2, status_score, vertical_threshold, Vertical_threshold):
        if 80 < angle1 < 100 and 80 < angle2 < 100:
            update_scores(status_score, 0.8, -0.035, 0)
            if vertical_threshold > Vertical_threshold:
                update_scores(status_score, 0.15, 0, 0)
        elif angle1 > 170 and 80 < angle2 < 100:
            update_scores(status_score, 0, 0.2, 0)
        elif angle1 < 90 and 0 < angle2 < 45:
            update_scores(status_score, 0, 0, 0.2)
        else:
            update_scores(status_score, 0, 0.05, 0.05)

# 调用处理两个不同的角度组合
    process_angles(Hip_Knee_Shoulders_angle, human_angle, status_score, vertical_threshold, Vertical_threshold)
    process_angles(Ankle_Knee_Hip_angle, human_angle, status_score, vertical_threshold, Vertical_threshold)


# # A肩部胯部膝盖夹角 
# # 角度在80度到100度之间，表明腿部与上半身有一定角度，类似坐下的姿势
# # 且人体整体角度在80度到100度之间，表示接近直立的状态
#     if 80 < Hip_Knee_Shoulders_angle < 100 and 80 < human_angle < 100:
#         status_score['Sit'] += 0.8
#         status_score['Stand'] += -0.035
#         #垂直方向参数超过预期，则进一步增加
#         if vertical_threshold > Vertical_threshold:
#             status_score['Sit'] += +0.15
#         _weight = f'{_weight}, [6]Stand:-0.035, Sit:+0.15'

#     elif Hip_Knee_Shoulders_angle > 170 and 80 < human_angle < 100:
#         status_score['Stand'] += 0.2

#     elif Hip_Knee_Shoulders_angle < 90 and 0 < human_angle < 45:
#         status_score['Fall'] += 0.2

#     else:
#         status_score['Fall'] += 0.05
#         status_score['Stand'] += 0.05
#         _weight = f'{_weight}, [7]Stand:+0.05, Fall:+0.05'


# # B胯部膝盖小腿
# # 角度在80度到100度之间，表明腿部与上半身有一定角度，通常表示膝盖和髋部弯曲，接近坐下
# # 且人体整体角度在80度到100度之间，表示接近直立的状态
#     if 80 < Ankle_Knee_Hip_angle < 145 and 80 < human_angle < 100:
#         status_score['Sit'] += 0.8
#         status_score['Stand'] += -0.035
#         if vertical_threshold > Vertical_threshold:
#             status_score['Sit'] += +0.15
#         _weight = f'{_weight}, [8]Stand:+0.035, Sit:+0.15'
#     else:
#         status_score['Fall'] += 0.05
#         status_score['Stand'] += 0.05
#         _weight = f'{_weight}, [9]Fall:+0.05, Stand:+0.05'

    # if 65 < Hip_Knee_Right_angle < 145 and 45 < human_angle < 125:
    #     status_score['Sit'] += 0.8
    #     if vertical_threshold > Vertical_threshold:
    #         status_score['Sit'] += +0.15
    # else:
    #     status_score['Fall'] += 0.05
    #     status_score['Stand'] += 0.05



    # status_score.values() 返回所有状态的分数。
    # status_score.keys() 返回所有状态的名称'Stand', 'Sit', 'Fall'
    # zip(status_score.values(), status_score.keys()) 将每个状态的分数与其名称配对
    # max() 函数在这些配对中找到分数最高的一个
    # 并返回对应的分数（score_max）及其状态名称（status_max）
    score_max, status_max = max(zip(status_score.values(), status_score.keys()))

    if status_max == 'Stand' or status_max == 'Sit' or status_max == 'other':
        status_max = 'Normal'
    else:
        status_max = 'Dangerous'

    return status_max
