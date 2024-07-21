# Write the requirements to a file
requirements = """
scikit-learn==1.0.2
tensorflow==2.9.1
keras==2.9.0
numpy==1.22.4
pandas==1.4.2
seaborn==0.11.2
matplotlib==3.5.2
warnings==0.1.2
"""

with open("requirements.txt", "w") as file:
    file.write(requirements)
