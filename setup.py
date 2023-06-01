from setuptools import setup, find_packages
from setuptools.command.install import install


class DownloadNLTK(install):
    def run(self):
        self.do_egg_install()
        import nltk
        nltk.download('punkt')


REQUIRES = """
transformers
accelerate
datasets>=2.7.1
evaluate>=0.3.0
faiss_gpu>=1.7.2
nltk>=3.8
numpy>=1.23.4
openai>=0.27.1
rank_bm25>=0.2.2
requests>=2.28.1
scikit_learn>=1.2.1
sentence_transformers>=2.2.2
torch>=1.13.1
tqdm>=4.64.1
"""


def get_install_requires():
    reqs = [req for req in REQUIRES.split("\n") if len(req) > 0]
    return reqs


with open("README.md") as f:
    readme = f.read()


def do_setup():
    setup(
        name="openicl",
        version='0.1.7',
        description="An open source framework for in-context learning.",
        url="https://github.com/Shark-NLP/OpenICL",
        author='Zhenyu Wu, Yaoxiang Wang, Zhiyong Wu, Jiacheng Ye',
        long_description=readme,
        long_description_content_type="text/markdown",
        cmdclass={'download_nltk': DownloadNLTK},
        install_requires=get_install_requires(),
        setup_requires=['nltk==3.8'],
        python_requires=">=3.8.0",
        packages=find_packages(
            exclude=[
                "test*",
                "paper_test*"
            ]
        ),
        keywords=["AI", "NLP", "in-context learning"],
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
        ]
    )


if __name__ == "__main__":
    do_setup()
