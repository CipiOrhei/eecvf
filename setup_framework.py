import subprocess

"""
Use this module to install all libraries and dependencies 
"""


__project_name__ = 'End-to-End CV Framework'
__cv_version__ = '0.1.dev1'
__author__ = 'Ciprian-Constantin ORHEI'

__install_list__ = 'requirements.txt'

if __name__ == '__main__':
    subprocess.call(['python', '-m', 'ensurepip', '--default-pip'])
    subprocess.call(['python', '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.call(['python', '-m', 'pip', 'install', 'numpy'])
    subprocess.call(['python', '-m', 'pip', 'install', '-r', __install_list__])
    subprocess.call(['python', '-m', 'pip', 'install', 'opencv-contrib-python-headless'])
