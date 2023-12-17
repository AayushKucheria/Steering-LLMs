import setuptools

setuptools.setup(
    name="Steering-LLMs",
    version="0.1.0",
    description="Steering of LLMs through addition of activation vectors with latent ethical valence",
    packages=setuptools.find_packages(where="src"),
    author="AI Safety Camp Project 16",
    author_email="",
    url="https://github.com/SkyeNygaard/Steering-LLMs",
    install_requires=[
        "transformers",
        "transformer_lens",
        "torch"
    ],
)
