from huggingface_hub import HfApi
import os

# Hosting script that can push all the deployment files into the Hugging Face space
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./predictive_maintenance/deployment",  # the local folder containing your files
    repo_id="subhradasgupta/predictive_maintenance",    # the target repo
    repo_type="space",                                  # dataset, model, or space
)
