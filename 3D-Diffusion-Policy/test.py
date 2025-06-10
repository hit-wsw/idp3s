import zarr
import numpy as np

def process_demos(zarr_path, num_demos=2):
    # 打开zarr文件
    root = zarr.open(zarr_path, mode='r')
    data = root['data']
    meta = root['meta']
    
    # 获取episode分界点
    episode_ends = meta['episode_ends'][:]
    demo_starts = np.concatenate(([0], episode_ends[:-1]))
    demo_ends = episode_ends
    
    # 随机选择2个demo
    selected_indices = np.random.choice(len(demo_starts), size=num_demos, replace=False)
    
    results = []
    for idx in selected_indices:
        start = demo_starts[idx]
        end = demo_ends[idx]
        demo_length = end - start
        
        # 提取demo数据
        demo_pcd = data['point_cloud'][start:end]
        demo_state = data['state'][start:end]
        
        # 处理每一帧
        demo_results = []
        for i in range(demo_length):
            # 处理最后帧的特殊情况
            if i == demo_length - 1:
                frame_dict = {
                    'point_cloud': np.stack([demo_pcd[i], demo_pcd[i]]),
                    'agent_pos': np.stack([demo_state[i], demo_state[i]]),
                }
            else:
                frame_dict = {
                    'point_cloud': np.stack([demo_pcd[i], demo_pcd[i+1]]),
                    'agent_pos': np.stack([demo_state[i], demo_state[i+1]]),
                }
            
            # 这里模拟调用动作生成函数
            # action = workspace.get_pose_from_policy(path, name, frame_dict)
            # 假设返回的动作我们暂时不处理
            
            print(frame_dict)
        

# 使用示例
zarr_path = "/media/wsw/SSD1T1/data/g1_sim_test_expert.zarr"
processed_demos = process_demos(zarr_path)