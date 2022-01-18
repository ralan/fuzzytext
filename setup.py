from setuptools import setup, find_packages


setup(
    name="fuzzytext",
    version="1.0.2",
    description="Framework for extracting structured data from fuzzy texts using the concept of similar contexts",
    author="Alen Rafagudinov",
    author_email="ralan@mail.ru",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/ralan/fuzzytext",
    keywords="NLP fuzzy extractor parser",
    install_requires=[
        "textdistance[extras]",
        "transformers",
    ],
    setup_requires=["flake8", "pytest-runner"],
    tests_require=["pytest"],
)
