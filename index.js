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



app.listen(3000, console.log("Server started on port 3000"));
