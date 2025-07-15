def main():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("kerrit/imagenet1kmediumtrain-100k")
    
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()
     