# Game Recommendation System

## Team Information
- **Course:** CMPSC/DS 410
- **Team Name:** Inspire
- **Members:**
  - Janvi Ahuja
  - John Xu
  - Umme Nujum
  - Zehao Liu
  - Shiquan Zhang
  - Sijie Yang
  - Sama Mehta
- **Instructor:** Romit Maulik
- **Teaching Assistants:**
  - Peng Jin
  - Haiwen Guan

## Project Overview
This project develops a game recommendation system using the dataset from Kaggle. The model's hyper-parameters were optimized using cross-validation, and the final model recommends games based on user preferences.

### Dataset
- **Source:** [Kaggle - Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?select=games.csv)

### Files Description
1. **project_test.ipynb**
   - **Purpose:** Finding best hyper-parameters.
   - **Details:** Parameters tested include k values of 4, 7, 10, 13 across separate jobs.
   - **Slurm Outputs:**
     - `slurm-10776966`: k = 4, regularization = 0.3, iterations = 30, Validation Error = 0.9450608986591409
     - `slurm-10776972`: k = 7, regularization = 0.3, iterations = 30, Validation Error = 0.9465092819687683
     - `slurm-10776977`: k = 10, regularization = 0.3, iterations = 30, Validation Error = 0.9462157986508757
     - `slurm-10776982`: k = 13, regularization = 0.3, iterations = 30, Validation Error = 0.9466413896353857
   - **Selected Parameters:** k = 4, regularization = 0.3, iterations = 30, Validation Error = 0.9450608986591409

2. **projectC.ipynb**
   - **Purpose:** Evaluates the validation RMS error and testing error using the chosen hyper-parameters.
   - **Slurm Output:**
     - `slurm-10785348`: Validation Error = 0.9437063160468666, Testing Error = 0.9442512290480936

3. **recommend.ipynb**
   - **Purpose:** Uses optimized parameters to train the ALS model, make predictions, and generate game recommendations for a specified user.
   - **Test User:** User ID 8762579
   - **Slurm Output:**
     - `slurm-10790243`: Top rated games include:
       - Game ID: 271590, Game Name: Grand Theft Auto V
       - Game ID: 1517290, Game Name: Battlefield™ 2042
       - Game ID: 1259420, Game Name: Days Gone
       - Game ID: 1029690, Game Name: Sniper Elite 5

## References
- **Relevant Labs:** CMPSC/DS 410 Lab6 and Lab7
