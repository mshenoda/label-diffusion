from setuptools import setup, find_packages

def get_requirements():
    with open("labeldiffusion/requirements.txt", "r") as requirements_file:
        requirements = requirements_file.readlines()
    return [req.strip() for req in requirements]

def get_version():
    with open("labeldiffusion/_version.py", "r") as version_file:
        version_code = version_file.read().strip()
        version = version_code.split('"')[1] if '__version__' in version_code else None
        return version

setup(
    name="labeldiffusion",
    version=get_version(),
    description="LabelDiffusion: Automatic Labeling of Stable Diffusion Pipelines",
    author="Michael Shenoda",
    license="GNU Affero General Public License (AGPL)",
    url="https://github.com/mshenoda/label-diffusion",
    author_email="info@mshenoda.com",
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=get_requirements()
)
