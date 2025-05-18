import visualizer
import h5py

def load_hdf5_data(path):
    with h5py.File(path, 'r') as file:
        data_0 = file['data']['demo_0']
        pointcloud_data = data_0['obs']['pointcloud'][0]
    return pointcloud_data
        
path = '/media/wsw/SSD1T1/data/g1_2actiongap_10000points.hdf5'

your_pointcloud = load_hdf5_data(path)
visualizer.visualize_pointcloud(your_pointcloud)