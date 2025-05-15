# AI Research Repository

This repository contains research papers, articles, and example code implementations on various topics in artificial intelligence and machine learning. The content is organized by topic, with each topic containing LaTeX source files, compiled PDFs, and example scripts demonstrating key concepts.

## Repository Structure

```
/
├── scripts/                # Utility scripts for the repository
│   └── latex_to_pdf.sh    # Script to convert LaTeX files to PDF
├── topics/                # Main content organized by topic
│   ├── contexts_and_chunking/
│   ├── reinforcement_learning/
│   ├── structured_llm_output/
│   └── tokenization_and_embedding/
└── requirements.txt       # Python dependencies for running example scripts
```

Each topic directory follows a standard structure:

```
/topics/topic_name/
├── latex/                 # LaTeX source files
├── example_scripts/       # Python implementations of key concepts
└── *.pdf                  # Compiled PDF articles
```

## Topics

### Contexts and Chunking

Explores methods for breaking down large texts into manageable chunks for processing by language models, including various chunking strategies and their relationship to context windows.

### Reinforcement Learning

Covers the fundamentals and advanced concepts in reinforcement learning:

- **Introduction to Reinforcement Learning**: Covers fundamental concepts including agent-environment interaction, Markov Decision Processes, policies, value functions, and classical algorithms like Q-learning.

- **Reinforcement Learning and LLMs**: Explores the intersection of RL and large language models, focusing on RLHF, RLAIF, and other techniques for aligning LLMs with human preferences.

- **State-of-the-Art RL Methods**: Examines cutting-edge approaches including evolutionary RL, meta-reinforcement learning, offline RL, and multi-agent systems.

Example scripts demonstrate implementations of Q-learning, Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Evolutionary Reinforcement Learning.

### Structured LLM Output

Focuses on techniques for generating and controlling structured outputs from large language models, including methods for ensuring consistent formatting and adherence to specific schemas.

### Tokenization and Embedding

Explores the process of converting text into numerical representations that can be processed by machine learning models, including various tokenization strategies and embedding techniques.

## Using This Repository

### Running Example Scripts

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run an example script:
   ```bash
   python topics/reinforcement_learning/example_scripts/basic_q_learning.py
   ```

### Generating PDFs from LaTeX

To generate a PDF from a LaTeX file, use the provided script:

```bash
bash scripts/latex_to_pdf.sh topics/topic_name/latex/file_name.tex
```

This will create a PDF in the topic's root directory.

## Contributing

Contributions to this repository are welcome. To contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Add your content following the established directory structure
4. Submit a pull request

## License

See the [LICENSE](LICENSE) file for details.
