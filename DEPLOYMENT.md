# Streamlit Cloud Deployment Guide

## Main File Path

For Streamlit Cloud deployment, use:

```
src/app.py
```

## Deployment Steps

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Connect Your GitHub Repository**:
   - Repository: `JeyanthPonnaluri/Mini_Project_RVR`
   - Branch: `main`
   - Main file path: `src/app.py`

3. **Python Version**: 
   - Python 3.9 or higher

4. **Requirements**:
   - The `requirements.txt` file is already in the repository root
   - All dependencies will be automatically installed

## Important Notes

### File Structure
```
Mini_Project_RVR/
├── src/
│   ├── app.py              ← Main Streamlit app (use this path)
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── logistic_numpy.py
│   ├── federated.py
│   ├── experiments.py
│   ├── sustainability.py
│   ├── fedprox_experiments.py
│   ├── contribution.py
│   ├── experiment_manager.py
│   └── ui_components.py
├── datasets/
├── reports/
├── requirements.txt
└── README.md
```

### Running Locally

To run the app locally from the repository root:

```bash
cd Mini_Project_RVR
python -m streamlit run src/app.py
```

Or from within the src directory:

```bash
cd Mini_Project_RVR/src
streamlit run app.py
```

### Data Upload

The app requires users to upload the clinical TSV file through the UI. The datasets are not included in the deployment for privacy and size reasons.

### Environment Variables

No environment variables are required for basic deployment.

## Troubleshooting

### Import Errors

If you encounter import errors, the app includes path handling:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

This ensures all modules in the `src` directory can be imported correctly.

### ModuleNotFoundError

If you see errors like `ModuleNotFoundError: No module named 'matplotlib'`:
1. Check that `requirements.txt` is in the repository root
2. Verify all dependencies are listed with proper versions
3. Wait for Streamlit Cloud to rebuild (it may take 2-3 minutes)
4. Check the deployment logs for any installation errors

Current dependencies:
- numpy>=1.24.0,<2.0.0
- pandas>=2.0.0,<3.0.0
- scikit-learn>=1.3.0,<2.0.0
- scipy>=1.10.0,<2.0.0
- matplotlib>=3.7.0,<4.0.0
- streamlit>=1.28.0
- typing-extensions>=4.5.0

### Memory Issues

The app processes large datasets. If you encounter memory issues on Streamlit Cloud:
- Consider using Streamlit Cloud's paid tier for more resources
- Or deploy on your own server with more RAM

### File Paths

All file paths in the app are relative and should work correctly in the deployment environment.

## Support

For issues, please open an issue on the GitHub repository:
https://github.com/JeyanthPonnaluri/Mini_Project_RVR/issues
