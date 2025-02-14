import cv2
import numpy as np

def stack_images_horizontal(images, max_width=1920):
    """Stack images horizontally with a maximum width"""
    total_width = sum(img.shape[1] for img in images)
    scale = min(1.0, max_width / total_width)
    
    if scale < 1.0:
        resized_images = [cv2.resize(img, (0,0), fx=scale, fy=scale) for img in images]
    else:
        resized_images = images
        
    return np.hstack(resized_images)

def create_batch_histogram_image(data_list, labels, num_bins=50, value_range=(-20, 20), width=1280, height=480):
    """Create a histogram visualization for multiple sets of data"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,165,0), (128,0,128)]  # Different colors for different batches
    
    # Calculate all histograms
    histograms = []
    max_count = 0
    for data in data_list:
        hist_counts, bin_edges = np.histogram(data, bins=num_bins, range=value_range)
        histograms.append(hist_counts)
        max_count = max(max_count, hist_counts.max())
    
    # Normalize and draw
    bin_width = int(width / num_bins)
    for i, hist_counts in enumerate(histograms):
        hist_counts = hist_counts / max_count * (height - 60)
        
        for j in range(num_bins):
            x1 = j * bin_width
            y1 = height - int(hist_counts[j]) - 30
            x2 = (j + 1) * bin_width - 1
            y2 = height - 30
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[i % len(colors)], 1)
    
    # Draw grid
    for i in range(0, num_bins, 5):
        x = i * bin_width
        cv2.line(img, (x, 0), (x, height-30), (200,200,200), 1)
    
    for i in range(5):
        y = int(i * (height-30) / 4)
        cv2.line(img, (0, y), (width, y), (200,200,200), 1)
    
    # Draw axes and labels
    cv2.line(img, (0, height-30), (width-1, height-30), (0,0,0), 2)
    cv2.line(img, (0, 0), (0, height-30), (0,0,0), 2)
    
    # Add x-axis labels
    for i in range(0, num_bins, 5):
        x_val = value_range[0] + (value_range[1] - value_range[0]) * i / num_bins
        x_pos = i * bin_width
        cv2.putText(img, f"{x_val:.1f}", (x_pos-20, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    # Add legend
    for i, label in enumerate(labels):
        y_pos = 20 + i * 20
        cv2.putText(img, label, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
    
    return img

def visualize_consensus_edges(img, kpts, edges, edge_scores, threshold=0.5):
    """Visualize edges with color based on consensus score"""
    img = img.copy()
    
    # Add legend
    cv2.putText(img, f"Static (<{threshold})", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(img, f"Dynamic (>={threshold})", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    # Draw edges
    for edge, score in zip(edges, edge_scores):
        a, b = edge
        color = (0,0,255) if score >= threshold else (0,255,0)
        pt1 = tuple(map(int, kpts[a]))
        pt2 = tuple(map(int, kpts[b]))
        cv2.line(img, pt1, pt2, color, 2)
        
        # Optionally draw score
        mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
        cv2.putText(img, f"{score:.2f}", mid_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img
