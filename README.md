<p align="center">
  <img src="https://img.icons8.com/material-outlined/96/cloud.png" width="100" alt="blue-cloud-image" style="filter: invert(32%) sepia(85%) saturate(3188%) hue-rotate(189deg) brightness(98%) contrast(95%);">
</p>

<p align="center">
    <h1 align="center">Podcast-AI</h1>
</p>

<p align="center">
    <em>Podcast-AI is a cutting-edge platform that leverages artificial intelligence to enhance podcast experiences. It automatically transcribes podcast episodes, generates summaries, and provides personalized recommendations, making it easier for users to discover and engage with content.</em>
</p>

<p align="center">
    <em>Developed with the software and tools below.</em>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/TypeScript-3178C6.svg?style=default&logo=TypeScript&logoColor=white" alt="TypeScript">
    <img src="https://img.shields.io/badge/React-61DAFB.svg?style=default&logo=React&logoColor=black" alt="React">
    <img src="https://img.shields.io/badge/Node.js-339933.svg?style=default&logo=Node.js&logoColor=white" alt="Node.js">
    <img src="https://img.shields.io/badge/Express.js-000000.svg?style=default&logo=Express&logoColor=white" alt="Express.js">
    <img src="https://img.shields.io/badge/GraphQL-E10098.svg?style=default&logo=GraphQL&logoColor=white" alt="GraphQL">
    <img src="https://img.shields.io/badge/SQLite-003B57.svg?style=default&logo=SQLite&logoColor=white" alt="SQLite">
    <img src="https://img.shields.io/badge/Docker-2496ED.svg?style=default&logo=Docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/VS%20Code-007ACC.svg?style=default&logo=Visual%20Studio%20Code&logoColor=white" alt="VS Code">
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <br>

  - [Overview](#overview)
  - [Features](#features)
  - [Repository Structure](#repository-structure)
  - [Modules](#modules)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
  - [User Guide](#user-guide)
    - [Frontend Setup](#frontend-setup)
    - [Backend Setup](#backend-setup)
  - [Development & Contribution](#development--contribution)
  - [License](#license)

</details>

<hr>

## Overview

Podcast-AI is an innovative platform that leverages the power of Artificial Intelligence to improve the podcast experience. It automatically transcribes episodes, provides summaries, and suggests content based on user preferences, enabling easier discovery and engagement.

---

## Features

- **Automated Transcriptions**: Converts podcast audio into accurate, searchable text.
- **Content Summarization**: Generates concise episode summaries for easy consumption.
- **Personalized Recommendations**: Suggests podcasts based on listening habits and preferences.
- **User-Friendly Interface**: Built with React for a seamless, interactive experience.
- **Efficient Data Storage**: SQLite for storing user data and preferences.
- **Scalable Backend**: Utilizes Node.js and Express for a robust and scalable backend.
- **Docker Support**: Easily deploy and scale the application using Docker.

---

## Repository Structure

```sh
└── podcast-ai
    ├── Audios
    ├── dataset
    │   └── wavs
    ├── schema.sql
    └── README.md
```

---

## Modules

<details closed><summary>Frontend</summary>

### Files and Descriptions

| File                                | Description                                                                                              | README |
|-------------------------------------|----------------------------------------------------------------------------------------------------------|--------|
| [AutoTranscript.ipynb](AutoTranscript.ipynb) | <code>► Main Jupyter Notebook that implements the auto-transcription logic using AI models.</code>        | [README](README.md) |

</details>

<details closed><summary>Backend</summary>

### Files and Descriptions

| File                                    | Description                                                                                             | README |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------|--------|

</details>

---

## Getting Started

### Installation
Got it! Here's the updated version reflecting the correct usage for your **AutoTranscript.ipynb** project, where users need to upload audio files and run parts of the notebook:

---

## Modules

### Files and Descriptions

| File                                    | Description                                                                                             | README |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------|--------|
| [AutoTranscript.ipynb](AutoTranscript.ipynb) | <code>► Jupyter notebook implementing the AI-based auto-transcription logic for podcasts.</code>         | [README](README.md) |

---

## Getting Started

### Installation

There are no external dependencies to install for this project. You can simply download or clone the repository to get started.

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/podcast-ai.git
   ```

2. **Upload Audio Files:**
   Place the audio files you want to transcribe into the `Audio` folder. The files should be in a supported format (e.g., MP3, WAV).

### Usage

Once the audio files are uploaded into the `Audio` folder:

1. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook and open `AutoTranscript.ipynb`.

   ```sh
   jupyter notebook AutoTranscript.ipynb
   ```

2. **Run the Notebook Cells:**
   Follow the steps in the notebook, executing each cell in order. The notebook is structured with clear instructions to guide you through the transcription process, from loading the audio files to generating the transcriptions.



## Development & Contribution

We welcome contributions! Please fork this repository and submit a pull request for any improvements or fixes. Make sure to follow our coding standards and include tests with your changes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
