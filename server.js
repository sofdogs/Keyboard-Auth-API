require("dotenv").config() // load .env variables
const express = require("express") // import express
const morgan = require("morgan") //import morgan
const {log} = require("mercedlogger") // import mercedlogger's log function
const cors = require("cors") // import cors
const UserRouter = require("./controllers/user") //import User Routes

//DESTRUCTURE ENV VARIABLES WITH DEFAULT VALUES
const {PORT = 4020} = process.env

// Create Application Object
const app = express()

// GLOBAL MIDDLEWARE
app.use(cors()) // add cors headers
app.use(morgan("tiny")) // log the request for debugging
app.use(express.json()) // parse json bodies

//used to access css pages
app.use(express.static('views'));

// ROUTES AND ROUTES
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/views/index.html');
});

app.use("./models/user", UserRouter) // send all "/user" requests to UserRouter for 

// APP LISTENER
app.listen(PORT, () => log.green("SERVER STATUS", `Listening on port ${PORT}`))


/*
const express = require('express');
const connectDB = require("./db");
const bodyParser = require("body-parser");
const app = express();
app.use(express.json())
const port = 4020;
const path = require('path');

//new authentication method
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));



//Connecting the Database
connectDB();


//use css files in views dir
app.use(express.static('views'));

// Define routes
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/views/welcome.html');
});

app.get('/login', (req, res) => {
  res.sendFile(__dirname + '/views/login.html');
});

//import route.js as middleware in server.js
app.use("/api/auth", require("./Auth/route"))

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
*/