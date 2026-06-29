# **🧠 Response Surface (AI-PRS) Analysis Tool**

This is a Streamlit web application designed for advanced data analysis, statistical modeling, and optimization using Response Surface Methodology (RSM). The tool allows users to upload experimental data, automatically generate predictive models, and use both traditional and AI-powered optimization algorithms to find the ideal input variables for desired outcomes.


## **✨ Key Features**

* **CSV Data Input**: Easily upload your experimental data in CSV format. The application intelligently parses variable descriptions, types, and ranges.  
* **Automated OLS Modeling**: Automatically performs Ordinary Least Squares (OLS) regression to generate second-order polynomial models for each dependent variable.  
* **Interactive Plotting**:  
  * Generate interactive 3D response surface plots to visualize the relationship between two independent variables and an outcome.  
  * Create 2D contour plots and trade-off analysis plots to compare different models.  
  * Overlay multiple models and actual data points for comprehensive visualization.  
* **Model Evaluation**: Assess model performance with "Actual vs. Predicted" scatter and bar plots.  
* **Multi-Objective Optimization**: Use traditional SciPy optimizers (SLSQP, SHGO, Basinhopping) to find optimal variable settings that satisfy single or multiple objectives.  
* **🤖 AI-Powered Optimization**: Leverage Bayesian Optimization (scikit-optimize) to intelligently and efficiently search for the best possible outcomes, even in complex search spaces.  
* **Variable Combination Analysis**: Automatically test and rank combinations of variables to identify which groups have the most significant impact on the outcome.  
* **🗑️ Drug Elimination Score**: Mathematically evaluate models to identify which drugs should be eliminated based on their toxicity to normal cells and lack of efficacy on cancer cells.
* **Report Generation**: Download detailed analysis and optimization reports in .docx format for documentation and sharing.

## **🚀 Getting Started**

Follow these instructions to set up and run the application on your local machine.

### **Prerequisites**

* Python 3.8 or newer.

### **Installation & Setup**

1. **Clone the repository:**  
   git clone \<your-repository-url\>  
   cd AI-PRS-webui

2. **Create and activate a virtual environment:** This isolates the project's dependencies from your system's Python installation.  
   * On **Windows**:  
     python \-m venv venv  
     venv\\Scripts\\activate

   * On **macOS/Linux**:  
     python3 \-m venv venv  
     source venv/bin/activate

3. **Install the required packages:**  
   pip install \-r requirements.txt

## **💻 Usage**

1. Launch the application:  
   Once the virtual environment is active and packages are installed, run the following command in your terminal:  
   streamlit run app.py

   The application will automatically open in a new tab in your web browser.  
2. **Workflow:**  
   * Use the sidebar to **upload your CSV data file**.  
   * Verify the **Output Folder Path**.  
   * Click the **"Run OLS Analysis"** button to generate the underlying models.  
   * Once the analysis is complete, use the main tabs to access the **Plotting Tools**, **Optimizer**, and **AI Optimizer**.

## **📁 File Structure**

The project is organized to separate UI, core logic, and utility functions for better maintainability.

The project is organized to separate UI, core logic, and utility functions for better maintainability.

Response-surface-optimization-tool/  
├── 📂 src/
│   ├── 📂 views/                \# Contains modules for each UI tab (e.g., plotting_view.py, elimination_view.py) 
│   ├── 📂 views_edu/            \# Contains simplified UI tabs for the Education Edition 
│   ├── 📂 logic/                \# Core data processing, modeling, and scientific features (e.g., models.py, drug_elimination.py)  
│   ├── 📂 utils/                \# Contains helper functions and session state management
│   ├── 📜 app.py                \# Main application file (UI orchestrator)  
│   └── 📜 app_edu.py            \# Main application file for Education Edition
├── 📜 requirements.txt       \# Project dependencies  
├── 📜 start.bat              \# Convenience script for Windows users  
├── 📜 start_edu.bat          \# Convenience script for Education Edition
└── 📜 README.md              \# You are here\!

## **🛠️ Technologies Used**

* **Application Framework:** [Streamlit](https://streamlit.io/)  
* **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)  
* **Statistical Modeling:** [Statsmodels](https://www.statsmodels.org/), [SciPy](https://scipy.org/)  
* **Machine Learning / AI:** [Scikit-learn](https://scikit-learn.org/), [Scikit-optimize](https://scikit-optimize.github.io/)  
* **Plotting:** [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)  
* **Report Generation:** [python-docx](https://python-docx.readthedocs.io/)

