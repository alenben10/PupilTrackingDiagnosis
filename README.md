# Eye Movement Evaluation for Multiple Sclerosis and Parkinson's Disease Using Computer Vision

## Overview
This project aims to develop a computer vision-based program to track and analyze eye movements in response to a moving stimulus on a screen. The objective is to differentiate between normal and abnormal eye movements, which can indicate the presence of neurodegenerative diseases such as Parkinson's Disease (PD) and Multiple Sclerosis (MS). The project is inspired by the standardized oculomotor and neuro-ophthalmic disorder assessment (SONDA) method detailed in the study "[Eye Movement Evaluation in Multiple Sclerosis and Parkinson's Disease Using a Standardized Oculomotor and Neuro-Ophthalmic Disorder Assessment](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2020.00971/full)".

## Background
Parkinson's Disease (PD) and Multiple Sclerosis (MS) are neurodegenerative disorders that significantly affect ocular mobility. PD is characterized by alterations in saccadic movements, while MS can cause internuclear ophthalmoplegia and nystagmus. Traditional diagnostic methods are often qualitative and subjective, and existing quantitative tests can be lengthy and exhausting for patients.

## Project Objectives
1. Develop a program that uses video input to track eye movements in response to a continuously moving stimulus on a screen.
2. Analyze the velocity and trajectory of the pupils to identify abnormal patterns indicative of PD and MS.
3. Differentiate between normal and abnormal eye movements to provide a preliminary diagnostic tool that is quick, non-invasive, and accessible for patient monitoring.

## Methodology
The project involves creating a system that allows eye movements to control a cursor on the screen, which follows a continuously moving saccadic stimulus (e.g., [example video](https://www.youtube.com/watch?v=Ihj2EddtKEw)). By capturing the coordinates of the cursor movement over time and comparing them to the position of the stimulus, we can calculate the velocity differences. Slow and erratic following speeds may indicate potential signs of PD or MS.

## Technology Stack
- **Mediapipe:** A framework for building multimodal applied machine learning pipelines.
- **OpenCV (CV2):** A library of programming functions mainly aimed at real-time computer vision.

## Implementation Steps
1. **Data Collection:** Gather video data of eye movements in response to the moving stimulus.
2. **Preprocessing:** Use Mediapipe and OpenCV to detect and track eye movements in the video data.
3. **Feature Extraction:** Extract features such as pupil coordinates, velocity, and trajectory from the tracked eye movements.
4. **Pattern Analysis:** Analyze the extracted features to identify patterns that differentiate normal from abnormal eye movements.
5. **Evaluation:** Validate the system's accuracy in detecting signs of PD and MS by comparing its output to established diagnostic methods.

## Expected Outcomes
The expected outcome of this project is a functional prototype that can track and analyze eye movements to provide preliminary diagnostic information. The system aims to offer a non-invasive, efficient, and accessible tool for the early detection and monitoring of neurodegenerative diseases.

## References
- [Eye Movement Evaluation in Multiple Sclerosis and Parkinson's Disease Using a Standardized Oculomotor and Neuro-Ophthalmic Disorder Assessment (SONDA)](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2020.00971/full)

## Installation
1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the program:
    ```bash
    python main.py
    ```

## Usage
- Prepare the video input for eye movement tracking.
- Run the program and follow the on-screen instructions to start tracking eye movements.
- Analyze the results to identify potential signs of PD or MS.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
