# Computer_Vision_Live_Emotion_Detection
**Motivation**

The motivation behind creating a Python computer vision model for emotion detection stemmed from the desire to provide companies with a valuable tool for enhancing various aspects of their operations, particularly during business meetings. By integrating this model, companies can effortlessly record video footage or enable live emotion detection during meetings, allowing them to gauge the overall mood and sentiment. This capability is invaluable for managers, as they can accurately assess the success of meetings based on the emotions detected. For instance, if clients exhibit signs of happiness, it indicates a positive outcome, whereas expressions of dissatisfaction may indicate areas for improvement. Additionally, managers can leverage the model to evaluate the performance of their team members, such as interns or analysts, by analyzing their emotional responses during interactions. Furthermore, companies can extend the utility of the model by integrating it into their existing CCTV systems, enabling real-time emotion detection among employees throughout the workday. Overall, this computer vision model serves as a versatile tool for enhancing decision-making processes, fostering better communication, and optimizing overall workplace dynamics.

**Methodology**

The methodology for developing the emotion detection model began with the acquisition of training and testing images, each categorized into seven emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised. These images were preprocessed to extract Histogram of Oriented Gradient (HOG) features, a method known for capturing local shape and gradient information. Subsequently, a pipeline was constructed, comprising feature scaling and a Random Forest classifier, to facilitate model training. Hyperparameter tuning via GridSearchCV was employed to optimize the model's performance.

The pivotal function, detect_emotion_in_video, was designed to analyze recorded video footage. This function utilizes a pre-trained Haar Cascade classifier to detect faces within the video frames. Once faces are identified, the model predicts the corresponding emotion using the extracted HOG features. Emotions are superimposed onto the video frames, providing real-time feedback on the emotional dynamics captured. This function serves as the cornerstone for potential integration into live video streams, enabling real-time emotion analysis during business meetings or surveillance scenarios. Ultimately, the goal is to enhance decision-making processes and optimize interpersonal interactions through the nuanced understanding of human emotions.

**Results**

Here's a demonstration showcasing the functionality of my function in action through a video. It effectively detects faces, encloses them within frames, and accurately labels them with corresponding emotions. In this particular video excerpt from the "Shark Tank" episode, you can observe the precise labeling of emotions. This capability opens avenues for prospective candidates to analyze the reactions of previous participants and tailor their strategies accordingly based on the emotions elicited from the judges during their proposals.

![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/1.png)
![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/2.png)
![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/3.png)
![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/4.png)
![github](https://github.com/pavelkimldn/Computer_Vision_Live_Emotion_Detection/blob/main/5.png)
