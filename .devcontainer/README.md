# Dev container configurations

This directory contains the configuration for dev containers, which is used to
initialize the development environment in **Codespaces**, **Visual Studio
Code**, and **JetBrains IDEs**. The environment is installed with all the
necessary dependencies for development and is ready for linting, formatting, and
running tests.

* **GitHub Codespaces**. Create a codespace for the repo by clicking
    the "Code" button on the main page of the repo, selecting the "Codespaces"
    tab, and clicking the "+". The configurations will automatically be used.
    Follow
    [this guide](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository)
    for more details.

* **Visual Studio Code**. Open the root folder of the repo in VS Code. A
    notification will pop up to open it in a dev container with the
    configuration. Follow
    [this guide](https://code.visualstudio.com/docs/devcontainers/tutorial)
    for more details.

* **JetBrains IDEs**. Open the `.devcontainer/devcontainer.json` in your
   JetBrains IDE. Click the docker icon to create a dev container.
   Follow
   [this guide](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html)
   for more details.