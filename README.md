# Oculography-Neuron-Network

---

## Annotation App

This application allows you to load either a **video** or a **directory of images**, navigate through the frames, and draw or fit ellipses on them. You can also adjust **gamma**, **contrast**, **ellipse transparency (alpha)**, and **ellipse edge transparency (edge alpha)** for display purposes. The edited frames can be saved along with binary masks and ellipse parameters (in a CSV file).

### Features

1. **Load Video or Images**  
   - Load a **video file** (MP4, AVI, etc.) or select a directory of **images** (PNG, JPG, etc.).
   - When images are loaded from a directory, they are sorted (using `natsort`) and treated like frames.

2. **Navigate Frames**  
   - Use the **slider** or the **< / >** buttons to move between frames.
   - Enter a frame number in the text field and press **Enter** to jump to that frame directly.

3. **Drawing Modes**  
   - **Draw Ellipse**: Click and drag on the image to define an ellipse, or click on existing ellipses to move or resize them.
   - **Draw Points**: Place multiple points with left-click, then click **Fit Ellipse** to create a best-fit ellipse around these points.

4. **Ellipse Interactions**  
   - **Resize** by dragging corner control points.  
   - **Rotate** by dragging the red rotation handle.  
   - **Move** the entire ellipse by dragging from inside its boundary.  
   - **Delete** the ellipse or points with the **Delete Ellipse/Points** button.

5. **Adjust Display Only**  
   - **Gamma Slider**: Adjust the gamma correction for display.  
   - **Contrast Slider**: Adjust the contrast for display.  
   - **Ellipse Interior Alpha Slider**: Control the transparency of the ellipse’s interior.  
   - **Ellipse Edge Alpha Slider**: Control the transparency of the ellipse’s outline (edge).  
   > **Note**: These display adjustments do not affect the saved images—they are only for visualization.

6. **Saving**  
   - Click **Save Frame** to store:
     - The **original frame** (no display adjustments applied).
     - A **binary mask** of the ellipse (if any).
     - Ellipse parameters saved to a **CSV** file, containing center, size, and angle for each saved frame.
   - If the **Go to next frame after saving** checkbox is checked, the application automatically advances to the next frame upon saving.

7. **Saved Annotations File Structure**  
    Annotations from this tool are saved in the following folder pattern. The **original name** is either the video name (plus its frame number) if a video was loaded, or the image name if a directory of images was loaded:
   
    ```txt
    Project Root
    └── annotated_data
        └── {Video or directory name}
            ├── annotations
            │   └── {Original name}.png
            ├── frames
            │   └── {Original name}.png
            └── metadata
                └── metadata.csv
    ```
   
---

## Video Frame Extractor

A Python application built with **PyQt5** and **OpenCV**, allowing you to load a video, preview it, select start and end frames, and extract frames (evenly spaced) from a chosen range. You can extract a specific number of frames or a certain percentage, optionally resizing them to a custom resolution.

### Features

1. **Load & Preview Video**  
   - Load common video formats such as `.mp4`, `.avi`, `.mov`, `.mkv` (any format OpenCV supports).  
   - Play, pause (toggles with the same button), or stop the video.  
   - Use a slider to seek through the video.

2. **Set Start and End Frames**  
   - Manually adjust start/end frame sliders or numeric spinboxes.  
   - Pause at a specific frame and click **"Set Start"** or **"Set End"** to snap to that frame.

3. **Select Frames to Extract**  
   - **By count**: Extract a fixed number of frames from the selected range.  
   - **By percentage**: Extract a certain percentage (1%–100%) of the range’s frames.

4. **Custom or Original Frame Size**  
   - If you want the frames in their original resolution, leave **"Custom frame size"** **unchecked**.  
   - To resize the extracted frames, check **"Custom frame size"** and enter your desired width and height.

5. **Progress Bar & Status**  
   - See how many frames have been extracted out of the total selected.  
   - A progress bar updates as each frame is processed.

6. **Video Information**  
   - Displays: total frames, FPS, and the original width × height of the video.  
   - Shows the current playback time and frame index.  
   - Automatically calculates how many frames will be extracted.

7. **Extracted Frames File Structure**

   Frames from Frame Extractor are saved following this pattern:

   ```txt
   Project Root
   └── extracted_frames
       └── {Video name}
           └── {Video name}_{Frame number}.png
