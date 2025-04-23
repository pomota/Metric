from setuptool import setup, find_packages

setup(
    name="report-metric",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "dspy",
        "python-dotenv",
        # 기타 필요한 패키지
    ],
    entry_points={
        'console_scripts': [
            'report-metric=metric.main:main',
        ],
    },
)