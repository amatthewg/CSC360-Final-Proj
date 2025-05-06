import kagglehub

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")

    print("Path to dataset files:", path)
