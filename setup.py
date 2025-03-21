from setuptools import find_packages, setup


hyphen_e_dot = '-e .'
def get_requirements(path):
    packages_to_download = []
    with open(path) as req:
        packages_to_download = req.readlines()
        packages_to_download = [package.replace('\n', '') for package in packages_to_download]

        if hyphen_e_dot in packages_to_download:
            packages_to_download.remove(hyphen_e_dot)

    return packages_to_download

setup(
    name='ML Project',
    version='0.0.1',
    author='Manjunath',
    author_email='manjunathyashram5@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)