
from setuptools import setup
from setuptools import find_packages

with open(file = "README.md", mode = "r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name             = 'facial_keypoint_detecter',
    version          = '1.0.0',
    author           = 'Shashank Kumbhare',
    author_email     = 'shashankkumbhare8@gmail.com',
    url              = 'https://github.com/ShashankKumbhare/facial-keypoint-detecter',
    description      = 'short discription short discription short discription',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license          = 'MIT',
    keywords         = ['python', 'opencv', 'ML', 'machine learning', 'facial-facial_keypoint_detecter-detecter',
                        'AI', 'artificial intelligence', 'xxxx'],
    install_requires = [
                        'numpy',
                        'matplotlib',
                        'scipy',
                        'IPython',
                        'opencv-python',
                       ],
    packages         = find_packages(),
    package_data     = {
                        'template_pkg' : ['__data_subpkg__/dataset_train/xxx/*',
                                          '__data_subpkg__/dataset_train/yyy/*',
                                          '__data_subpkg__/dataset_train/zzz/*',
                                          '__data_subpkg__/dataset_test/XXX/*',
                                          '__data_subpkg__/dataset_test/YYY/*',
                                          '__data_subpkg__/dataset_test/ZZZ/*',
                                          '__auxil_subpkg__/images/*',
                                          '__constants_subpkg__/*',]
                       },
    include_package_data = True,
    classifiers      = ['License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3'
                       ]
)
