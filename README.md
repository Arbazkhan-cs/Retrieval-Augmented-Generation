---
## PDF Question Answering System

### Overview
The PDF Question Answering System is a web application that allows users to upload PDF documents and ask questions about their content. The system extracts text from the PDFs, processes it, and leverages state-of-the-art language models to provide accurate and context-aware answers to user queries.
https://huggingface.co/spaces/Arbazkhan-cs/Retrieval-Augmented-Generation

### Features
- **PDF Text Extraction:** Utilizes PyPDF2 to extract text from uploaded PDF documents.
- **Text Chunking:** Employs RecursiveCharacterTextSplitter to split text into manageable chunks based on newline characters.
- **Vector Store Indexing:** Uses FAISS for efficient text indexing and retrieval.
- **Embeddings Generation:** Leverages HuggingFace's sentence-transformers for creating embeddings.
- **Natural Language Processing:** Integrates the ChatGroq large language model (LLM) to handle and respond to natural language queries.
- **User-Friendly Interface:** Built with Streamlit for easy interaction and query submission.

### How It Works
1. **Upload a PDF:** Users upload a PDF document through the Streamlit interface.
2. **Text Processing:** The application extracts and splits the text into chunks.
3. **Indexing:** The text chunks are indexed using FAISS.
4. **Query Submission:** Users input questions related to the PDF content.
5. **Answer Retrieval:** The system retrieves the most relevant text chunks and generates answers using the LLM.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Installation
1. **Clone the Repository:**
    ```sh
    git clone https://github.com/Arbazkhan-cs/Retrieval-Augmented-Generation.git
    cd your-repo
    ```

2. **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

Then, run the app using the following command:
```sh
streamlit run app.py
```

## Technologies Used
- **Languages and Frameworks:** Python, Streamlit
- **Libraries:** PyPDF2, langchain, sentence-transformers, FAISS, dotenv
- **Models and Tools:** Hugging Face LLMs, ChatGroq, RecursiveCharacterTextSplitter

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries, please contact [your email](mailto:your.email@example.com).

---

Feel free to customize the text to better fit your project's specifics and your preferences.
