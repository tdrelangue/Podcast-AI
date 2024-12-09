<p align="center">
  <img src="https://img.icons8.com/color/microphone" width="100" alt="blue-cloud-image" style="filter: invert(32%) sepia(85%) saturate(3188%) hue-rotate(189deg) brightness(98%) contrast(95%);">
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
  <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python"> 
  <img src="https://img.shields.io/badge/Jupyter%20Notebook-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter Notebook"> 
  <img src="https://img.shields.io/badge/Alive%20Progress-0A7D8C.svg?style=flat&logo=alive-progress&logoColor=white" alt="Alive Progress"> 
  <img src="https://img.shields.io/badge/Whisper-2A9DF4.svg?style=flat&logo=whisper&logoColor=white" alt="Whisper"> 
  <img src="https://img.shields.io/badge/Torch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white" alt="Torch"> 
  <img src="https://img.shields.io/badge/Lightning-FF6F00.svg?style=flat&logo=pytorch-lightning&logoColor=white" alt="Lightning"> 
  <img src="https://img.shields.io/badge/Torchaudio-25A2F1.svg?style=flat&logo=torchaudio&logoColor=white" alt="Torchaudio"> 
  <img src="https://img.shields.io/badge/Torchvision-05A5D1.svg?style=flat&logo=opencv&logoColor=white" alt="Torchvision"> 
  <img src="https://img.shields.io/badge/Pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas"> 
  <img src="https://img.shields.io/badge/PyOpenCL-11A44D.svg?style=flat&logo=opencl&logoColor=white" alt="PyOpenCL"> 
  <img src="https://img.shields.io/badge/Icecream-4B5B57.svg?style=flat&logo=icecream&logoColor=white" alt="Icecream"> </p>

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
