

## Generating Data Using YouTube videos

this script (imgscraper.py) collects frames of interest from YouTube (using FFMPEG), and automatically masks them (using Meta's Segment Anything Model), hopefully to generate datapoints that we can use to train our YOLO model.

Future students should focus on refining and improving the segmentation method, so less "junk" data is included in the output, and finding ways to store the training data (and preventing the same videos from being scraped multiple times unnecessarily).

**Contribute by adding videos of teardowns/replacements [here](https://www.youtube.com/playlist?list=PLdDkfYYZfGNYC5-HmpokPXi3U1OYrf-U1&jct=Bj-L04zOa5GJwlBIWZlR_g)**


### Requirements:
- [SAM model](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth)
- [pytubefix](https://pytubefix.readthedocs.io/en/latest/user/install.html)
- [cv2](https://pypi.org/project/opencv-python/)
- [ffmpeg-python](https://pypi.org/project/ffmpeg-python/)

### Example Output:

<img width="609" height="1237" alt="iPhone_14_Battery_Dead_Replace_it_in_No_Time_frame_0017_masked" src="https://github.com/user-attachments/assets/eb6fc5c5-6b94-4045-ba19-6a68a2f1666f" />

<img width="489" height="955" alt="iPhone_11_Battery_Replacement_Fix_A_Dead_Or_Dying_Battery_frame_0039_masked" src="https://github.com/user-attachments/assets/e2ea6938-ea7c-4033-8528-88038c437745" />


*ideal output. Hands, and other obstructions are cropped out.*

<img width="2850" height="1633" alt="iPhone_14_Battery_Dead_Replace_it_in_No_Time_frame_0010_masked" src="https://github.com/user-attachments/assets/6c95c99a-2eff-4bdc-a5e2-4be3e8e87ecf" />


*This was part of the output from the script. Ideally, the script shouldn't output this. Still could be useful though.*
