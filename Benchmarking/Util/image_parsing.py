import os


def find_img_extension(path):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.raw']

    for extension in extensions:
        if os.path.exists(os.path.join(path + extension)):
            return extension
    else:
        return extensions[0]


if __name__ == "__main__":
    pass
