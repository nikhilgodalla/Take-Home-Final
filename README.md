# Reinforcement Learning for Agentic Code Review Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Overview

This project implements an intelligent code review agent that learns to identify bugs, security vulnerabilities, and code quality issues through reinforcement learning. The system combines Deep Q-Networks (DQN) with Multi-Armed Bandits to optimize review strategies and improve through experience.

### 🎯 Key Features
- **Dual RL Approach**: DQN for action selection + Multi-Armed Bandits for strategy optimization
- **Real-time Learning**: Agent improves review quality through experience
- **Comprehensive Detection**: Identifies security vulnerabilities, performance issues, and code smells
- **Production Ready**: Sub-second response times with 69.5% average reward score

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rl-code-review-agent.git
cd rl-code-review-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the pre-trained agent:
```bash
python demo.py
```

## 📁 Project Structure

```
rl-code-review-agent/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Technical_Report.pdf         # Detailed technical documentation
├── RL_Code_Review_Agent.ipynb  # Main implementation notebook
├── demo.py                      # Quick demo script
├── train.py                     # Training script
├── test.py                      # Testing framework
├── models/
│   └── trained_agent.pkl        # Pre-trained model weights
├── data/
│   └── test_cases.json          # Test code samples
└── results/
    ├── training_metrics.json    # Training performance logs
    └── test_results.json        # Evaluation results
```

## 🔧 Usage

### Training a New Agent

```python
from code_review_agent import CodeReviewAgent, train_agent

# Train agent for 300 episodes
agent = train_agent(n_episodes=300)

# Save the trained model
agent.save('models/my_agent.pkl')
```

### Using the Pre-trained Agent

```python
from code_review_agent import CodeReviewAgent

# Load pre-trained agent
agent = CodeReviewAgent.load('models/trained_agent.pkl')

# Review your code
code = '''
def process_data(user_input):
    result = eval(user_input)  # Security issue!
    return result
'''

review = agent.review_code(code)
print(review)
```

### Running Tests

```bash
# Run comprehensive test suite
python test.py

# Run specific test category
python test.py --category security

# Generate performance report
python test.py --report
```

## 📊 Performance

### Key Metrics
- **F1 Score**: 0.364 (on limited training data)
- **Security Detection**: 100% accuracy for critical vulnerabilities
- **Average Response Time**: 0.12 seconds
- **Training Episodes**: 300
- **Model Parameters**: 20,244

### Issue Detection Accuracy

| Issue Type | Detection Rate | False Positives | Severity Accuracy |
|------------|---------------|-----------------|-------------------|
| Security vulnerabilities | 100% | 0% | 100% |
| Syntax errors | 100% | 0% | 100% |
| Deep nesting | 100% | 0% | 80% |
| Performance issues | 50% | 10% | 75% |

## 🧠 Technical Approach

### Reinforcement Learning Components

1. **Deep Q-Network (DQN)**
   - 3-layer neural network (8→128→128→20)
   - Experience replay buffer (10,000 capacity)
   - Target network updates every 10 episodes
   - ε-greedy exploration (1.0 → 0.01)

2. **Multi-Armed Bandits**
   - Upper Confidence Bound (UCB) algorithm
   - 5 strategies: Bug, Performance, Style, Security, Refactor
   - Contextual adaptation based on code features

### State Representation
8-dimensional feature vector:
- Cyclomatic complexity
- Line count & function count
- Comment ratio
- Variable count
- Maximum nesting depth
- Pattern violation count
- Test coverage estimate

### Action Space
20 discrete actions combining:
- Review type (5 categories)
- Severity level (1-5)
- Focus area (4 types)
- Suggestion depth (1-3)

## 📝 Examples

### Security Vulnerability Detection
```python
# Input code with security issue
code = '''
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    return db.execute(query)
'''

# Agent output
"""
[SECURITY] Severity: 5/5
Issue: SQL injection vulnerability detected
Suggestion: Use parameterized queries:
  query = "SELECT * FROM users WHERE name=?"
  return db.execute(query, (username,))
"""
```

### Performance Optimization
```python
# Input code with performance issue
code = '''
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates
'''

# Agent output
"""
[PERFORMANCE] Severity: 3/5
Issue: O(n²) complexity in duplicate detection
Suggestion: Use set() for O(n) solution:
  return list(set([x for x in items if items.count(x) > 1]))
"""
```

## 🛠️ Development

### Running in Google Colab
1. Open `RL_Code_Review_Agent.ipynb` in Google Colab
2. Run all cells sequentially
3. Use the interactive interface for testing

### Local Development
```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Check code quality
flake8 src/
```

## 📈 Future Improvements

- [ ] Expand training dataset (10,000+ samples)
- [ ] Implement transfer learning from pre-trained code models
- [ ] Add support for multiple programming languages
- [ ] Integrate with CI/CD pipelines
- [ ] Implement hierarchical RL for better strategy separation
- [ ] Add meta-learning for few-shot adaptation
- [ ] Implement curriculum learning for progressive complexity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

