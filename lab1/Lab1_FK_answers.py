import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


class Stack:

    def __init__(self):
        self.data = []

    def push(self, data):
        self.data.append(data)

    def pop(self):
        return self.data.pop()

    def top(self):
        return self.data[len(self.data) - 1]


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    f = open(bvh_file_path)
    lines = f.readlines()
    joint_name = []
    joint_parent = []
    joint_offset = []

    joint_stack = Stack()  # 栈中存着树的层级结构，如[-1 0 1 2]代表已经处理到ROOT为树根底下第三个节点
    joint_stack.push(-1)  # 每个节点都有唯一编号，其中ROOT为-1
    index = 0

    for line in lines:
        words = line.split()
        line_first_word = words[0]
        if line_first_word == "ROOT" or line_first_word == "JOINT":
            joint_name.append(words[1])
        elif line_first_word == "OFFSET":
            joint_offset.append(
                [float(words[1]),
                 float(words[2]),
                 float(words[3])])
        elif line_first_word == "End":
            joint_name.append(joint_name[joint_stack.top()] + "_end")
        elif line_first_word == "{":  # 读入{时入栈，开始处理新关节
            joint_parent.append(joint_stack.top())  # 新节点的父亲为栈顶元素
            joint_stack.push(index)
            index += 1
        elif line_first_word == "}":  # 读入}时出栈
            joint_stack.pop()
        elif line_first_word == "MOTION":
            break

    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset,
                             motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    pose = motion_data[frame_id]  # 某帧的动作数据
    joint_positions = []
    joint_orientations = []
    joint_local_orientations = []
    # ROOT数据
    joint_positions.append(np.array(pose[0:3]))
    joint_orientations.append(R.from_euler(
        'XYZ', np.array(pose[3:6]), degrees=True))
    joint_local_orientations.append(R.from_euler(
        'XYZ', np.array(pose[3:6]), degrees=True))

    inner_node_index = 1  # 内部节点（叶子节点以外）index
    for i in range(1, len(joint_name)):
        parent_index = joint_parent[i]  # 当前节点的父亲
        if "_end" in joint_name[i]:  # 对于叶子(外部)节点，其局部朝向==I（无子节点,不存在朝向问题）
            joint_local_orientations.append(R.identity())
            joint_global_rotation = joint_orientations[parent_index]
        else:  # 对于一般(内部)节点，其局部朝向==pose
            joint_local_orientations.append(R.from_euler('XYZ',
                                                         pose[3 * (inner_node_index+1):3 * (
                                                             inner_node_index+2)],
                                                         degrees=True))
            # O[i]=O[p]*R[i]
            joint_global_rotation = joint_orientations[parent_index] * \
                joint_local_orientations[i]
            inner_node_index += 1

        joint_global_position = joint_positions[parent_index] + \
            joint_orientations[parent_index].apply(joint_offset[i])
        joint_positions.append(joint_global_position)
        joint_orientations.append(joint_global_rotation)

    for i in range(len(joint_orientations)):
        joint_orientations[i] = joint_orientations[i].as_quat()  # 转化为四元数表示

    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出:
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    motion_data = []
    A_motion_data = load_motion_data(A_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(
        A_pose_bvh_path)
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(
        T_pose_bvh_path)

    for A_pose in A_motion_data:
        T_pose = [A_pose[0], A_pose[1], A_pose[2]]

        inner_node_index = 1
        T_pose_pre = []  # 预处理数据
        for A_name in A_joint_name:
            if "_end" in A_name:
                T_pose_pre.append([0, 0, 0])
            else:
                T_pose_pre.append(
                    A_pose[3*inner_node_index:3*(inner_node_index+1)])
                inner_node_index += 1

        inner_node_index = 1
        for T_name in T_joint_name:
            j = 0
            for A_name in A_joint_name:
                if T_name == A_name:
                    A_index = j
                    break
                j += 1

            if "_end"not in T_name:
                T_pose.extend(T_pose_pre[A_index])
                if "lShoulder" in T_name:
                    T_pose[3*inner_node_index+2] -= 45
                if "rShoulder" in T_name:
                    T_pose[3*inner_node_index+2] += 45
                inner_node_index += 1
        motion_data.append(T_pose)

    return np.array(motion_data)
