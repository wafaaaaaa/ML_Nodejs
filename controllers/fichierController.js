const express = require("express");
var router = express.Router();

const multer = require("multer");
var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, process.env.destinationimportcollab);
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
    //Appending extension
  },
});
var upload = multer({ storage: storage });
//installation pip dotenv
//upload csv
router.post(
  "/add-collabs-from-file",
  upload.single("add_collabs_file"),
  async (req, res) => {
    res.json({ file: req.file });
  }
);

router.post("/resetdata", upload.single("add_collabs_file"), (req, res) => {
  const childPython = require("child_process").spawn(process.env.python, [
    "Prediction_semaine.py",
    req.body.nom,
  ]);

 console.log('params')

 var dataToSend;
 childPython.stdout.on("data", function (data) {
   dataToSend = `${data}`;
   console.log(dataToSend);
   res.send(dataToSend);
 });

 childPython.on("close", (code) => {
   // res.send(dataToSend);
 })

   childPython.stderr.on("data", (data) => {
    //console.error(`stderr: ${data}`);
  });
 
  // childPython.on("close", (code) => {
  //   console.log(`child process exited with code ${code}`);
  // });
});

router.post("/resetdatamois", upload.single("add_collabs_file"), (req, res) => {
  const childPython = require("child_process").spawn(process.env.python, [
    "Prediction_mois.py",
    req.body.nom,
  ]);

 console.log('params')

 var dataToSend;
 childPython.stdout.on("data", function (data) {
   dataToSend = `${data}`;
   console.log(dataToSend);
   res.send(dataToSend);
 });

 childPython.on("close", (code) => {
   // res.send(dataToSend);
 })

   childPython.stderr.on("data", (data) => {
    //console.error(`stderr: ${data}`);
  });
 
  // childPython.on("close", (code) => {
  //   console.log(`child process exited with code ${code}`);
  // });
});
module.exports = router;
