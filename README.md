# Financial Analyst Agent

This project implements a Financial Analyst Agent using LangChain and Google Generative AI. The agent can fetch financial data using `yfinance` and perform web searches using `tavily-python` to provide comprehensive financial analysis.

<img width="3418" height="1880" alt="image" src="https://github.com/user-attachments/assets/218860b5-d910-4357-b8ae-6dd49a96ff31" />
<img width="1828" height="1398" alt="image" src="https://github.com/user-attachments/assets/25b4b879-18b7-43ab-b5bb-84f43e11a1c7" />
<img width="1846" height="1888" alt="image" src="https://github.com/user-attachments/assets/ec88ff24-504c-459c-adcd-b427a84715c2" />

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Orange443/financial-analyst-agent.git
cd financial-analyst-agent
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the root directory of the project and add your API keys:

```
GOOGLE_API_KEY="your_google_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

Replace `your_google_api_key_here` with your actual Google API key and `your_tavily_api_key_here` with your actual Tavily API key.

### 5. Run the Application

To start the Financial Analyst Agent, run the `main.py` file:

```bash
python main.py
```

Follow the prompts in the terminal to interact with the agent.
