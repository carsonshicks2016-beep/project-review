import os
import sys
import glob
import cv2
import numpy as np

def analyze_screenshots(directory, output_path):
    print(f"Analyzing screenshots in {directory}...")
    screenshots = glob.glob(os.path.join(directory, "*.png"))
    if not screenshots:
        print("No screenshots found.")
        return []

    anomalies = []

    # Fog / Sky color is approx (166, 187, 194) in RGB, which is (194, 187, 166) in BGR (OpenCV)
    # Let's define a range around it for the sky/void color.
    lower_bound = np.array([170, 160, 140])
    upper_bound = np.array([220, 210, 190])

    for filepath in sorted(screenshots):
        img = cv2.imread(filepath)
        if img is None:
            continue
        
        h, w, _ = img.shape
        # Only look at the bottom 40% of the screen where ground should be
        # If there's sky color here, it's a void/hole.
        bottom_region = img[int(h * 0.6):h, :]
        
        mask = cv2.inRange(bottom_region, lower_bound, upper_bound)
        
        # Find contours of the void color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        if contours:
            max_area = max(cv2.contourArea(c) for c in contours)
        
        # If the largest contiguous void blob is bigger than 2% of the bottom region, flag it
        region_area = bottom_region.shape[0] * bottom_region.shape[1]
        if max_area > (region_area * 0.02):
            filename = os.path.basename(filepath)
            anomalies.append({
                "file": filename,
                "path": filepath,
                "void_area_pct": (max_area / region_area) * 100
            })
            print(f"FLAGGED: {filename} - Void detected ({max_area/region_area*100:.1f}% of ground)")
            
            # Draw rectangle on the image for the report
            # Find the largest contour again
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            # Adjust y for the cropped region
            y += int(h * 0.6)
            
            annotated = img.copy()
            cv2.rectangle(annotated, (x, y), (x+cw, y+ch), (0, 0, 255), 3)
            cv2.putText(annotated, f"VOID DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            report_dir = os.path.dirname(os.path.abspath(output_path))
            annotated_path = os.path.join(report_dir, filename.replace(".png", "_flagged.png"))
            cv2.imwrite(annotated_path, annotated)
            anomalies[-1]["annotated_path"] = annotated_path
        else:
            print(f"PASSED: {os.path.basename(filepath)}")

    return anomalies

def generate_report(anomalies, output_path):
    with open(output_path, "w") as f:
        f.write("# Visual QA Anomaly Report\n\n")
        
        if not anomalies:
            f.write("✅ **No severe visual anomalies (white voids/missing terrain) were detected!**\n")
            f.write("The procedural culling fixes appear to have successfully stabilized the geometry.\n")
            return
        
        f.write("❌ **Visual Anomalies Detected**\n\n")
        f.write(f"Found {len(anomalies)} screenshots with significant clipping/voids in the ground area.\n\n")
        
        f.write("## Details\n\n")
        for a in anomalies:
            f.write(f"### {a['file']}\n")
            f.write(f"- **Void Coverage**: {a['void_area_pct']:.1f}% of the ground region\n")
            # We copy the annotated image to the artifacts directory in the bash script
            # Here we just link to it assuming it's in the same directory as the report
            img_name = os.path.basename(a['annotated_path'])
            f.write(f"![{a['file']}]({img_name})\n\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing screenshots")
    parser.add_argument("--report", required=True, help="Path to output markdown report")
    args = parser.parse_args()
    
    anomalies = analyze_screenshots(args.dir, args.report)
    generate_report(anomalies, args.report)
    print(f"Report generated at {args.report}")
