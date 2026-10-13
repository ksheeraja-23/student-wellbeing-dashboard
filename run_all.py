"""One command to rebuild everything: python run_all.py"""
import subprocess, sys
steps = [[sys.executable, "-m", "src.preprocess"], [sys.executable, "export_figures.py"], [sys.executable, "build_notebook.py"],
         [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
          "student_wellbeing_analysis.ipynb", "--ExecutePreprocessor.timeout=600"]]
for s in steps:
    print(">>", " ".join(s)); subprocess.run(s, check=True)
print("\nDone. Launch the dashboard with:  streamlit run app.py")
