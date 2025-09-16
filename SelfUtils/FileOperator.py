import os
from SelfUtils.Logger import Logger
from PIL import Image
from typing import List, Tuple
import shutil

class FileOperator:
    @staticmethod
    def delete_invalid_files(folder_path: str, logger: Logger = None, keep_extensions=None):
        """
        Delete files in the specified folder that do not have the specified extensions.
        :param folder_path: The path to the folder containing files.
        :param logger: Logger instance for logging messages. If None, a default logger will be used.
        :param keep_extensions: List of file extensions to keep. Defaults to common image formats.
        """
        if keep_extensions is None:
            keep_extensions = ['.jpg', '.jpeg', '.png']
        deleted_count = 0
        for filename in os.listdir(folder_path):
            if not any(filename.endswith(ext) for ext in keep_extensions):
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {str(e)}")
        logger.info(f"Deleted {deleted_count} invalid files in {folder_path}")
        return len(os.listdir(folder_path))

    @staticmethod
    def batch_rename_files(folder_path: str, prefix: str = 'image_', start_index: int = 1, logger: Logger = None):
        """
        Batch rename files in the specified folder with a given prefix and starting index.
        :param folder_path: The path to the folder containing files.
        :param prefix: The prefix to use for renaming files.
        :param start_index: The starting index for renaming.
        :param logger: Logger instance for logging messages. If None, a default logger will be used.
        """
        files = os.listdir(folder_path)
        for i, filename in enumerate(files, start=start_index):
            old_file_path = os.path.join(folder_path, filename)
            new_file_name = f"{prefix}{i}{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(folder_path, new_file_name)
            try:
                os.rename(old_file_path, new_file_path)
                if logger:
                    logger.info(f"Renamed {old_file_path} to {new_file_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to rename {old_file_path} to {new_file_path}: {str(e)}")

    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                return text
        except FileNotFoundError:
            print(f"错误：找不到文件 '{file_path}'")
        except Exception as e:
            print(f"错误：无法读取文件 '{file_path}': {e}")
        return None

    @staticmethod
    def remove_files_not_in_list(folder_path: str, file_list: List[str]) -> None:
        """
        Remove files in the specified folder that are not in the provided file list.
        :param folder_path: The path to the folder containing files.
        :param file_list: List of filenames to keep in the folder.
        :return: None
        """
        try:
            if not os.path.exists(folder_path):
                print(f"Folder '{folder_path}' does not exist!")
                return

            if not os.path.isdir(folder_path):
                return

            current_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            files_to_delete = [f for f in current_files if f not in file_list]

            for file in files_to_delete:
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)

        except Exception as e:
            print(f"Something going wrong: {e}")

    @staticmethod
    def get_files_in_folder(image_dir: str) -> list[str]:
        """
        Get a list of files in the specified folder.
        Args:
            image_dir (str): The path to the folder.
        Returns:
            list[str]: A list of file names in the folder.
        """
        try:
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"文件夹不存在: {image_dir}")
            if not os.path.isdir(image_dir):
                raise NotADirectoryError(f"不是有效文件夹: {image_dir}")

            return [image_dir + '/' + name for name in os.listdir(image_dir)
                    if os.path.isfile(os.path.join(image_dir, name))]
        except Exception as e:
            print(f"获取文件列表时出错: {e}")
            return []

    @staticmethod
    def load_images(folder_path: str) -> Tuple[List[Image.Image], List[str]]:
        """
        Load images from a specified folder.

        :param folder_path: Path to the folder containing images.
        :return: A tuple of lists containing loaded images and their corresponding names.
        """
        images = []
        image_names = []

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    image = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                    images.append(image)
                    image_names.append(filename)
                except Exception as e:
                    print(f"Skipping {filename}: {e}")
        return images, image_names

    @staticmethod
    def copy_files_to_folder(file_paths: List[str], target_folder: str) -> None:
        """
        Copy files from a list of file paths to the target folder.

        :param file_paths: List of file paths
        :param target_folder: Target folder path
        """
        # Ensure the target folder exists
        os.makedirs(target_folder, exist_ok=True)

        for file_path in file_paths:
            if os.path.isfile(file_path):
                try:
                    shutil.copy2(file_path, target_folder)
                    print(f"Copied: {file_path} -> {target_folder}")
                except Exception as e:
                    print(f"Failed to copy {file_path}: {e}")
            else:
                print(f"File does not exist, skipped: {file_path}")

    @staticmethod
    def copy_and_rename_files(file_paths: List[str], target_folder: str, new_names: List[str]) -> None:
        """
        Copy files from a list of file paths to the target folder
        and rename them using the provided list of new names.

        :param file_paths: List of file paths
        :param target_folder: Target folder path
        :param new_names: List of new names (same length as file_paths)
        """
        if len(file_paths) != len(new_names):
            raise ValueError("The length of new_names must match the length of file_paths.")

        # Ensure the target folder exists
        os.makedirs(target_folder, exist_ok=True)

        for file_path, new_name in zip(file_paths, new_names):
            if os.path.isfile(file_path):
                try:
                    # Get file extension
                    ext = os.path.splitext(file_path)[1]
                    new_file_name = f"{new_name}{ext}"
                    new_file_path = os.path.join(target_folder, new_file_name)

                    # If file with the same name already exists, add suffix
                    counter = 1
                    while os.path.exists(new_file_path):
                        new_file_name = f"{new_name}_{counter}{ext}"
                        new_file_path = os.path.join(target_folder, new_file_name)
                        counter += 1

                    # Copy and rename
                    shutil.copy2(file_path, new_file_path)
                    print(f"Copied and renamed: {file_path} -> {new_file_path}")
                except Exception as e:
                    print(f"Failed to copy {file_path}: {e}")
            else:
                print(f"File does not exist, skipped: {file_path}")