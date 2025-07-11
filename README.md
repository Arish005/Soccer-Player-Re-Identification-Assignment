# Player and Referee Detection with Stable ID Assignment

This repository presents a solution for detecting and uniquely identifying players and referees from a football match video using a YOLOv11-based model. It has two options for identification and tracking:
- **Option 1 Cross-Camera Player Mapping.
- **Option 2 Re-Identification in a Single Feed
---

## Setup Instructions

### Step 1: I have mannually added the files into the repository and for the google colab notebook into the repository I have added by Saving a copy in GitHub repository.```

### Step 2: Install Dependencies (in Google Colab or local)
```python
!pip install ultralytics opencv-python-headless gdown scikit-learn --quiet
!apt-get -y install ffmpeg  
```

### Step 3: Run the Code
Run the Colab notebook or `main.py` script to process the video.

---

## Dependencies & Environment

| Tool         | Version 	  |
|--------------|------------------|
| Python       | 3.8+    	  |
| Ultralytics  | Latest  	  |
| OpenCV       | 4.x     	  | 
| Scikit-learn | 1.0+    	  |
| FFmpeg       | System-installed |
| Google Colab | (Recommended)    |

---

## Report: Approach & Methodology

### Objective
- Detect football players and referees from a match video.
- Track them across frames while keeping their ID stable.

### Techniques Used

| Task 		   | Method 						|
|------------------|----------------------------------------------------|
| Object Detection | YOLOv11 pretrained model 				|
| ID Matching 	   | Cosine similarity of visual features 		|
| Filtering 	   | HSV-based green field detection to eliminate crowd |
| Performance 	   | Frame-by-frame processing and video generation 	|

### Techniques Tried & Results
- Used green HSV thresholds to reduce outside detections 
- Feature vector-based cosine similarity for identity assignment 
- Feature extraction using resized crops (64x128) 

### Challenges Encountered
- Frame-level identity stability with slight pose changes
- Referees in blue getting low similarity scores in poor lighting
- Processing time due to full-frame detection + similarity check

### Future Improvements
- Train a lightweight classifier for ID matching
- Use better embedding extractor like MobileNet/ReID models
- Incorporate motion tracking (Kalman filter or DeepSORT)
- Run on GPU for faster inference

---

## Output
The processed video with bounding boxes and stable IDs is saved as:

```
videos/output_fixed.mp4
```

This file shows:
- Players with consistent IDs (green boxes)
- Referees tracked with correct IDs (blue boxes)
- Crowd and outside elements filtered out

## Required Files (Auto Downloaded)

The following files are automatically downloaded using `gdown` during execution.  
However, for transparency, here are the direct Google Drive links:

| File                  	      | Google Drive Link                                                                    |
|-------------------------------------|--------------------------------------------------------------------------------------|
| `15sec_input_720p.mp4` (Video)      | https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link|
| `yolo11_players.pt` (YOLOv11 Model) | https://drive.google.com/file/d/1TdeefIxQIPoiZeVV7VvFW7998_iFRyTl/view?usp=drive_link|
| `tacticam.mp4`		      | https://drive.google.com/file/d/1lU4t6OdtVHGAda1oE9PNH_hcctBqLuDS/view?usp=drive_link|
| `broadcast.mp4`		      | https://drive.google.com/file/d/1S9nRMF-rCc1EspNZ0qk8fTmWEZ9_uDs0/view?usp=drive_link|

 
> These files are automatically downloaded in the code using `gdown`.  
> No need for manual download unless running without internet.


---

## Performance Note

> **This notebook may take longer to execute** due to:
> - High-resolution video
> - Real-time detection + cosine similarity matching
>
> Optimization options like frame resizing and caching are noted in the code but kept commented to preserve accuracy.

---


```
├── models/
│ └── yolo11_players.pt # Pretrained YOLOv11 model
├── videos/
│ ├── broadcast.mp4 # Broadcast video input
│ ├── tacticam.mp4 # Tacticam video input
│ ├── output.mp4 # Output with raw bounding boxes
│ └── output_fixed.mp4 # Final encoded video output
├── Task_Option_1_Cross_Camera_Player_Mapping_.ipynb
├── Task_Option_2_Re_Identification_in_a_Single_Feed.ipynb
└── README.md

              
```

---

## Final Remarks

> This project demonstrates real-time computer vision + ID tracking using pretrained models and classic similarity techniques. It handles complex frames, filters out background, and provides a smooth detection pipeline ready to scale.
