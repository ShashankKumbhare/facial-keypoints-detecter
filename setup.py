
from setuptools import setup
from setuptools import find_packages

with open(file = "README.md", mode = "r") as readme_handle:
    long_description = readme_handle.read()

setup(
    name             = 'facial_keypoints_detecter',
    version          = '1.0.0',
    author           = 'Shashank Kumbhare',
    author_email     = 'shashankkumbhare8@gmail.com',
    url              = 'https://github.com/ShashankKumbhare/facial-keypoints-detecter',
    description      = 'A CNN based facial keypoints detector.',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license          = 'MIT',
    keywords         = ['facial-keypoints-detecter', 'facial landmarks', 'facial filters', 'PyTorch',
                        'torchvision', 'OpenCV-Python', 'python', 'ML', 'machine learning', 'AI',
                        'artificial intelligence'],
    install_requires = [
                        'numpy',
                        'pandas',
                        'matplotlib',
                        'opencv-python'
                       ],
    packages         = find_packages(),
    package_data     = {
                        'facial_keypoints_detecter' : [ '__data_subpkg__/keypoints_frames_test.csv',
                                                        '__data_subpkg__/keypoints_frames_train.csv',
                                                        '__constants_subpkg__/detector_architectures/*',
                                                        '__constants_subpkg__/filters/*',
                                                        '__constants_subpkg__/saved_models/YYY/*',
                                                        '__auxil_subpkg__/images/*',
                                                        '__constants_subpkg__/*' ]
                       },
    include_package_data = True,
    classifiers      = ['License :: OSI Approved :: MIT License',
                        'Natural Language :: English',
                        'Operating System :: OS Independent',
                        'Programming Language :: Python :: 3'
                       ]
)
