# **Automated Student Grading System**

## **1. Objective**
This project aims to develop an AI-powered system that automates and enhances student grading accuracy. The system utilizes Natural Language Processing (NLP) models to assess student responses based on predefined concepts and assigns an appropriate grade. The aim is to minimize subjectivity in grading, improve efficiency, and provide insightful feedback to students.

## **2. Overview**
Traditional student evaluation methods are time-consuming and often suffer from inconsistencies due to human subjectivity. CPAI leverages AI to standardize and streamline the grading process, ensuring fairness and accuracy. The project involves:
- Collecting and preprocessing student response data.
- Developing a robust NLP model for text classification.
- Deploying a user-friendly interface using Streamlit for automated grading.
- Evaluating the model's performance and accuracy in predicting grades.

## **3. Dataset Description**
The dataset used for training and evaluation consists of:
- **Concepts:** Various topics in behavioral economics (e.g., loss aversion, endowment effect, prospect theory).
- **Student Responses:** Textual answers given by students in assessments.
- **Labels:** Assigned grades based on predefined grading criteria (e.g., A+, A, B, C, D, F).

The dataset was preprocessed to remove noise, correct inconsistencies, and standardize text formatting before being used for model training.

## **4. Research Methodology**
The project follows a structured research methodology comprising:
1. **Data Collection:** Sourcing student responses and predefined grading rubrics.
2. **Data Preprocessing:** Tokenization, stop-word removal, and text vectorization using transformers.
3. **Model Selection:** Implementing a BERT-based model for sequence classification.
4. **Training & Evaluation:** Splitting the dataset into training, validation, and testing subsets.
5. **Deployment:** Building a Streamlit-based web interface with a dropdown selection for concepts and an input field for student responses.
6. **Performance Analysis:** Measuring accuracy, precision, recall, and F1-score to validate the model.

## **5. Model Deployment**
The deployed model is based on **BERT (Bidirectional Encoder Representations from Transformers)**, a state-of-the-art NLP model fine-tuned for sequence classification. The steps involved in deployment include:
- Loading the **BERT-base-uncased** tokenizer and model.
- Configuring the model with **six grading categories**.
- Implementing a **dropdown feature** in Streamlit for concept selection.
- Processing the student response and generating a grade prediction.
- Displaying the predicted grade dynamically through the UI.

## **6. Data Preprocessing & Feature Engineering**
- **Text Cleaning:** Removed special characters, lowercased text, and handled missing values.
- **Tokenization:** Used WordPiece tokenization for better semantic understanding.
- **Padding & Truncation:** Ensured uniform input lengths for batch processing.
- **Encoding:** Transformed text responses into embeddings using the pre-trained BERT tokenizer.

## **7. Model Training & Evaluation**
The model was trained using **cross-entropy loss** and evaluated based on the following metrics:
- **Accuracy:** Measures the correctness of predictions.
- **Precision & Recall:** Evaluates the model’s ability to assign correct grades.
- **Confusion Matrix:** Analyzes misclassification patterns.
- **F1-Score:** Balances precision and recall for overall performance assessment.

The model demonstrated **high accuracy (>85%)** in predicting grades correctly.

## **8. Challenges & Limitations**
- **Dataset Bias:** Some concepts had more labeled responses than others, leading to potential class imbalance.
- **Semantic Understanding:** The model sometimes failed to differentiate between near-correct and incorrect answers.
- **Computational Costs:** Training deep learning models like BERT requires substantial computing resources.

## **9. Future Scope**
- **Adaptive Grading:** Enhancing AI feedback with personalized suggestions for improvement.
- **Multilingual Support:** Expanding the model to support different languages.
- **Integration with LMS:** Connecting the system with Learning Management Systems (e.g., Moodle, Blackboard) for seamless grading.
- **Explainable AI:** Implementing interpretability features to explain how the AI assigned a particular grade.

## **10. Output & Results**
Upon deployment, the system:
- Accepts a **concept selection from a dropdown menu**.
- Takes a **student’s response as input**.
- Processes the input through the fine-tuned **BERT model**.
- Outputs a **predicted grade** along with a confidence score.
- It provides **real-time results** through a user-friendly Streamlit interface.

## **11. Conclusion**
Ths project successfully demonstrates how AI can revolutionize student assessment by making grading **faster, more consistent, and data-driven**. With further improvements, the system can be scaled for broader applications in educational institutions worldwide.


