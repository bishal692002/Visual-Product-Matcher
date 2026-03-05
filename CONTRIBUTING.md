# Contributing to Visual Product Matcher

Thank you for your interest in contributing! Contributions of all kinds are welcome — bug fixes, new features, documentation improvements, and more.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Visual-Product-Matcher.git
   cd Visual-Product-Matcher
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Development Workflow

1. Create a new branch for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, keeping commits small and focused.
3. Test your changes manually by running the app:
   ```bash
   streamlit run app.py
   ```
4. Push your branch and open a Pull Request against `main`.

---

## Pull Request Guidelines

- Keep the scope of each PR focused — one feature or bug fix per PR.
- Write a clear PR title and description explaining **what** changed and **why**.
- Reference any related issues using `Closes #<issue-number>`.
- Ensure your code passes any existing tests before submitting.

---

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/bishal692002/Visual-Product-Matcher/issues) and include:

- A clear, descriptive title
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Python version, OS, and relevant package versions

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use descriptive variable and function names.
- Add docstrings to all public functions.
- Keep lines under 100 characters where practical.

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
