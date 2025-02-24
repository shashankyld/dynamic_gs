from core.map import Map
from utilities.utils_draw import visualize_global_map, visualize_local_map

if __name__ == '__main__':
    map_path = "/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/logs/slam_map/20250224_134722/map.pkl"
    try:
        print("Loading map from:", map_path)
        loaded_map = Map.load(map_path)
        
        # Print map statistics
        print("\nMap Statistics:")
        print(f"Number of keyframes: {len(loaded_map.keyframes)}")
        print(f"Number of map points: {len(loaded_map.map_points)}")
        print(f"Number of local map points: {len(loaded_map.local_map_points)}")
        print(f"Number of local keyframes: {len(loaded_map.local_keyframes)}")
        
        # Visualize the loaded map
        print("\nVisualizing map...")
        visualize_global_map(loaded_map, 
                             title="Loaded Map Visualization",
                             dense=True)
        visualize_local_map(loaded_map,
                            title="Loaded Local Map Visualization",
                            dense=True)
        
    except Exception as e:
        print(f"Error loading map: {e}")
