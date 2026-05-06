import streamlit as st
import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt
from collections import deque
import tempfile
from PIL import Image

# -------------------- CAR CLASS --------------------
class Car:
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.frames_crossed = 0
        self.cross_start_frame = None
        self.cross_end_frame = None

    def getId(self):
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn
        self.frames_crossed += 1

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_UP(self, mid_start, mid_end):
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end:
                self.state = '1'
                self.dir = 'up'
                return True
        return False

    def going_DOWN(self, mid_start, mid_end):
        if len(self.tracks) >= 2 and self.state == '0':
            if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start:
                self.state = '1'
                self.dir = 'down'
                return True
        return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True

# -------------------- UTILITY FUNCTIONS --------------------
def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def estimate_aqi(car_count, truck_count):
    car_emission_rate = 120
    truck_emission_rate = 900

    total_car_emissions = car_count * car_emission_rate
    total_truck_emissions = truck_count * truck_emission_rate
    total_emissions = total_car_emissions + total_truck_emissions

    aqi = min(int(total_emissions / 100), 500)
    if aqi < 50:
        return (aqi, "Good", "green")
    elif aqi < 100:
        return (aqi, "Moderate", "yellow")
    elif aqi < 150:
        return (aqi, "Unhealthy for Sensitive Groups", "orange")
    elif aqi < 200:
        return (aqi, "Unhealthy", "red")
    elif aqi < 300:
        return (aqi, "Very Unhealthy", "purple")
    else:
        return (aqi, "Hazardous", "maroon")

def process_video(video_path, speed_limit, progress_bar):
    # Initialize variables
    os.makedirs('detected', exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=200, varThreshold=90)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kernelOp = np.ones((3, 3), np.uint8)
    kernelCl = np.ones((11, 11), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cars = []
    max_p_age = 5
    pid = 1

    cnt_up = 0
    cnt_down = 0
    cnt_car = 0
    cnt_truck = 0

    line_up = 400
    line_down = 250
    up_limit = 230
    down_limit = int(4.5 * (500 / 5))

    real_distance_meters = 8.0  # Estimated real-world distance between lines
    pixel_distance = abs(line_down - line_up)
    meters_per_pixel = real_distance_meters / pixel_distance
    overspeed_count = 0
    overspeeding_ids = []
    speeds = {}
    
    aqi_history = []
    frame_count = 0
    
    # For visualization
    frames = []
    bar_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        
        frame = cv2.resize(frame, (900, 500))
        fgmask = fgbg.apply(frame)

        ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        contours0, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_centroids = []

        for cnt in contours0:
            area = cv2.contourArea(cnt)
            if area > 300:
                m = cv2.moments(cnt)
                if m['m00'] == 0:
                    continue
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                x, y, w, h = cv2.boundingRect(cnt)

                if up_limit < cy < down_limit:
                    detected_centroids.append((cx, cy, x, y, w, h))

        for cx, cy, x, y, w, h in detected_centroids:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            matched = False
            for car in cars:
                if euclidean_distance(cx, cy, car.getX(), car.getY()) < 50:
                    car.updateCoords(cx, cy)
                    matched = True

                    if car.going_UP(line_down, line_up) and not hasattr(car, 'counted'):
                        cnt_up += 1
                        aspect_ratio = w / float(h)
                        area = w * h
                        label = "Truck" if area > 30000 or (area > 5000 and aspect_ratio < 1.2) else "Car"
                        cnt_car += 1 if label == "Car" else 0
                        cnt_truck += 1 if label == "Truck" else 0
                        car.counted = True

                        time_seconds = car.frames_crossed / fps
                        real_distance = pixel_distance * meters_per_pixel
                        speed = (real_distance / time_seconds) * 3.6
                        if 0 < speed < 180:
                            speeds[car.getId()] = round(speed, 2)
                            if speed > speed_limit:
                                overspeed_count += 1
                                overspeeding_ids.append(car.getId())
                                vehicle_img = frame[y:y+h, x:x+w]
                                filename = f"detected/overspeed_vehicle_{car.getId()}.jpg"
                                cv2.imwrite(filename, vehicle_img)

                    elif car.going_DOWN(line_down, line_up) and not hasattr(car, 'counted'):
                        cnt_down += 1
                        aspect_ratio = w / float(h)
                        area = w * h
                        label = "Truck" if area > 30000 or (area > 5000 and aspect_ratio < 1.2) else "Car"
                        cnt_car += 1 if label == "Car" else 0
                        cnt_truck += 1 if label == "Truck" else 0
                        car.counted = True
                        time_seconds = car.frames_crossed / fps
                        real_distance = pixel_distance * meters_per_pixel
                        speed = (real_distance / time_seconds) * 3.6
                        if 0 < speed < 180:
                            speeds[car.getId()] = round(speed, 2)
                            if speed > speed_limit:
                                overspeed_count += 1
                                overspeeding_ids.append(car.getId())
                                vehicle_img = frame[y:y+h, x:x+w]
                                filename = f"detected/overspeed_vehicle_{car.getId()}.jpg"
                                cv2.imwrite(filename, vehicle_img)
                    break

            if not matched:
                new_car = Car(pid, cx, cy, max_p_age)
                cars.append(new_car)
                pid += 1

        for car in cars[:]:
            car.age_one()
            if car.timedOut():
                cars.remove(car)

        frame = cv2.line(frame, (0, line_up), (900, line_up), (255, 0, 255), 2)
        frame = cv2.line(frame, (0, up_limit), (900, up_limit), (0, 255, 255), 2)
        frame = cv2.line(frame, (0, down_limit), (900, down_limit), (255, 0, 0), 2)
        frame = cv2.line(frame, (0, line_down), (900, line_down), (255, 0, 0), 2)

        cv2.putText(frame, f'UP: {cnt_up}', (10, 40), font, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'DOWN: {cnt_down}', (10, 80), font, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f'Cars: {cnt_car}', (10, 120), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Trucks: {cnt_truck}', (10, 160), font, 0.6, (0, 255, 255), 2)
        
        # Store frame for display
        frames.append(frame)
        
        # Calculate AQI and store data for graphs
        estimated_aqi, _, _ = estimate_aqi(cnt_car, cnt_truck)
        aqi_history.append(estimated_aqi)
        bar_data.append([cnt_car, cnt_truck, cnt_up, cnt_down])
    
    cap.release()
    
    # Create graphs
    # Bar chart
    fig_bar, ax_bar = plt.subplots()
    categories = ['Cars', 'Trucks', 'Up', 'Down']
    values = [cnt_car, cnt_truck, cnt_up, cnt_down]
    ax_bar.bar(categories, values)
    ax_bar.set_title('Vehicle Counts')
    ax_bar.set_ylabel('Count')
    
    # AQI line graph
    fig_aqi, ax_aqi = plt.subplots()
    ax_aqi.plot(range(len(aqi_history)), aqi_history, 'r-')
    ax_aqi.set_title('AQI Over Time')
    ax_aqi.set_xlabel('Frame Number')
    ax_aqi.set_ylabel('AQI')
    
    return {
        'frames': frames,
        'bar_chart': fig_bar,
        'aqi_chart': fig_aqi,
        'stats': {
            'total_vehicles': cnt_up + cnt_down,
            'cars': cnt_car,
            'trucks': cnt_truck,
            'up': cnt_up,
            'down': cnt_down,
            'overspeeding': overspeed_count,
            'speeds': speeds,
            'overspeeding_ids': overspeeding_ids,
            'aqi': estimate_aqi(cnt_car, cnt_truck)
        }
    }

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(page_title="Traffic Analysis System", page_icon="🚦", layout="wide")
    
    st.title("🚗 Traffic Analysis and Environmental Impact System")
    st.markdown("""
    This system analyzes traffic patterns, calculates vehicle speeds, and estimates the environmental impact 
    based on vehicle counts. It can:
    - Count vehicles moving in each direction
    - Classify vehicles as cars or trucks
    - Detect speeding vehicles
    - Calculate Air Quality Index (AQI) based on traffic
    """)
    
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
    speed_limit = st.sidebar.slider("Speed Limit (km/h)", 30, 120, 60)
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        st.sidebar.success("File uploaded successfully!")
        
        # Process video
        progress_bar = st.progress(0)
        with st.spinner('Processing video... This may take a few minutes depending on video length'):
            results = process_video(tfile.name, speed_limit, progress_bar)
        progress_bar.empty()
        
        # Display results
        st.success("Processing complete!")
        
        # Show video with annotations
        st.header("Processed Video")
        video_placeholder = st.empty()
        for frame in results['frames']:
            video_placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.03)  # Control playback speed
        
        # Display statistics
        st.header("Traffic Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Vehicles", results['stats']['total_vehicles'])
        col2.metric("Cars", results['stats']['cars'])
        col3.metric("Trucks", results['stats']['trucks'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Vehicles Going Up", results['stats']['up'])
        col2.metric("Vehicles Going Down", results['stats']['down'])
        col3.metric("Overspeeding Vehicles", results['stats']['overspeeding'])
        
        # Display AQI
        aqi, aqi_label, aqi_color = results['stats']['aqi']
        st.header("Environmental Impact")
        st.markdown(f"""
        <div style="background-color:{aqi_color};padding:10px;border-radius:5px;">
            <h3 style="color:white;">Air Quality Index: {aqi} ({aqi_label})</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display charts
        st.header("Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(results['bar_chart'])
        with col2:
            st.pyplot(results['aqi_chart'])
        
        # Display speed data
        st.header("Speed Analysis")
        if results['stats']['speeds']:
            st.write("Vehicle speeds (km/h):")
            for vid, spd in results['stats']['speeds'].items():
                if spd > speed_limit:
                    st.error(f"Vehicle {vid}: {spd} km/h (OVERSPEEDING)")
                else:
                    st.info(f"Vehicle {vid}: {spd} km/h")
        else:
            st.warning("No speed data collected - check your detection lines")
        
        # Clean up
        os.unlink(tfile.name)
    else:
        st.info("Please upload a traffic video to get started")

if __name__ == "__main__":
    main()