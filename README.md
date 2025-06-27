# LabChronicle_streamlit | Lab Database Interface

This repository contains the interface for our lab's reagent management platform.

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### Prerequisites

* Python 3.8+
* [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
* `git` (for cloning repositories)

### Installation

1.  **Clone this repository:**

    ```bash
    git clone [https://github.com/yourusername/labchronicle.git](https://github.com/yourusername/labchronicle.git)
    cd labchronicle
    ```

2.  **Create a Conda environment (recommended):**

    ```bash
    conda create -n labchronicle_env python=3.9 # Or your preferred Python version
    conda activate labchronicle_env
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *(To generate `requirements.txt` from your current environment, run `pip freeze > requirements.txt`)*

### Running the Application

1.  **Activate your Conda environment:**

    ```bash
    conda activate labchronicle_env
    ```

2.  **Start the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

3.  Your browser should automatically open to the application (usually at `http://localhost:8501`).

## 👨‍💻 Usage

The application provides an interface to interact with our lab's data management system. Follow the on-screen prompts to load data and explore the available information.

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for any suggestions or improvements.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

