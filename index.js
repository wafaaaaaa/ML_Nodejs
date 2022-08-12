const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");
const bodyParser = require("body-parser");

require("dotenv").config();
var fichierController = require("./controllers/fichierController.js");

const app = express();

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors({ origin: process.env.cors }));
app.use("/api", fichierController);

//   #importation des bibliothèques necéssaires
// import pandas as pd
// import numpy as np
// import seaborn as sns
// import matplotlib.pyplot as plt
// import plotly.graph_objects as go
// from google.colab import drive
// drive.mount('/content/drive')
// data_16xls = pd.read_excel('/content/drive/MyDrive/2016.xlsx', dtype=str, index_col=None)
// data_16xls.to_csv('data16.csv', encoding='utf-8', index=False)
// df16 = pd.read_csv("data16.csv")
// #Add New Column
// df1['Nombre de pics'] = df1['Quantité']/df1['Unité/Pack']
//   dataFrame=dataFrame.drop(labels='DocumentNo', axis=1)
// dataFrame=dataFrame.drop(labels='LineNo', axis=1)
// dataFrame=dataFrame.drop(labels='Type', axis=1)
// dataFrame=dataFrame.drop(labels='GrossWeight', axis=1)
// dataFrame=dataFrame.drop(labels='gesamt nett', axis=1)
// dataFrame=dataFrame.drop(labels='gesamt brutt', axis=1)
// dataFrame=dataFrame.drop(labels='number of position', axis=1)
// dataFrame=dataFrame.drop(labels='Ligne ordre', axis=1)

app.listen(3000, console.log("Server started on port 3000"));
